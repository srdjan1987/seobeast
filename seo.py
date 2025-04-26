import webview
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import nltk
from openai import OpenAI
import json
from urllib.parse import urlparse
import language_tool_python

nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
kw_model = KeyBERT()

# 🔑 Your Real API Keys
API_KEY = "API_KEY"
CX = "CX KEY"
OPENAI_KEY = "OPEN_AI_KEY"
headers = {"User-Agent": "Mozilla/5.0"}

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

# Initialize language tool for grammar checking
language_tool = language_tool_python.LanguageTool('en-US')

class Api:
    def __init__(self):
        self.openai_client = client

    def get_google_results(self, query, country_code, limit=5):
        gl_map = {
            "Australia": "AU",
            "United States": "US",
            "Canada": "CA",
            "United Kingdom": "GB"
        }
        gl_param = gl_map.get(country_code, "US")
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX}&num={limit}&gl={gl_param}&hl=en"
        res = requests.get(url)
        items = res.json().get("items", [])
        return [item['link'] for item in items]

    def extract_content(self, url):
        """Extract content from URL with timeout"""
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.text
        except requests.Timeout:
            raise Exception("Request timed out. Please check your internet connection.")
        except requests.RequestException as e:
            raise Exception(f"Error fetching URL: {str(e)}")

    def extract_brand_name(self, url):
        """Extract brand name from domain"""
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            # Remove TLD and split remaining parts
            brand = domain.split('.')[0]
            # Clean up common business identifiers
            brand = brand.replace('-', ' ').replace('_', ' ')
            brand = ' '.join(word.capitalize() for word in brand.split())
            return brand
        except:
            return None

    def analyse_content(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # Basic stats
        stats = {
            "word_count": len(text.split()),
            "h1": len(soup.find_all('h1')),
            "h2": len(soup.find_all('h2')),
            "h3": len(soup.find_all('h3')),
            "images": len(soup.find_all('img')),
            "bold": len(soup.find_all(['b', 'strong'])),
            "internal_links": len([a for a in soup.find_all('a') if a.get('href', '').startswith('/')]),
            "paragraphs": len(soup.find_all('p')),
            "lists": len(soup.find_all(['ul', 'ol'])),
            "meta_description": None,
            "meta_title": None
        }
        
        # Extract meta information
        meta_desc = soup.find('meta', {'name': 'description'})
        meta_title = soup.find('title')
        stats["meta_description"] = meta_desc.get('content') if meta_desc else None
        stats["meta_title"] = meta_title.string if meta_title else None
        
        return stats

    def tfidf_keywords(self, base, competitors):
        docs = [base] + competitors
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(docs)
        words = TfidfVectorizer().fit(docs).get_feature_names_out()
        return sorted(zip(words, tfidf.toarray()[0]), key=lambda x: -x[1])[:20]

    def extract_lsi_keywords(self, text):
        return kw_model.extract_keywords(text, top_n=15, stop_words='english')

    def expand_lsi_keywords(self, text):
        return kw_model.extract_keywords(text, top_n=50, stop_words='english')

    def score_content(self, html, keywords):
        score = 0
        reasons = []
        hidden_deductions = 0
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            total_words = len(text.split())
            keyword_presence = self.analyze_keyword_presence(html, keywords)
            
            # First paragraph check (15 points)
            if keyword_presence['first_para']:
                score += 15
            reasons.append("✅ Main keyword in first paragraph" if keyword_presence['first_para'] else "❌ Add main keyword to first paragraph")
            
            # Heading distribution check (20 points max)
            heading_count = len(keyword_presence['headings'])
            heading_score = min(heading_count * 5, 20)
            score += heading_score
            reasons.append(f"✅ Main keyword in {heading_count} headings" if heading_count > 0 else "❌ Add main keyword to headings")
            
            # Keyword density check (15 points)
            density = keyword_presence['density']
            if 1 <= density <= 3:
                score += 15
            reasons.append(f"✅ Optimal keyword density ({density:.1f}%)" if 1 <= density <= 3 else 
                         f"{'❌' if density > 3 else '⚠️'} Keyword density is {density:.1f}% (aim for 1-3%)")
            
            # Content formatting checks (20 points)
            # Bold text check
            bold_elements = soup.find_all(['strong', 'b'])
            if bold_elements:
                score += 7
            reasons.append("✅ Content uses bold formatting" if bold_elements else "❌ Add bold formatting for emphasis")
            
            # Images check
            images = soup.find_all('img')
            if images:
                score += 7
            reasons.append(f"✅ Content includes {len(images)} images" if images else "❌ Add relevant images")
            
            # Lists check
            lists = soup.find_all(['ul', 'ol'])
            if lists:
                score += 6
            reasons.append("✅ Content includes lists" if lists else "❌ Add bullet points or numbered lists")
            
            # Content length check (15 points)
            if total_words >= 1500:
                score += 15
            elif total_words >= 800:
                score += 8
            reasons.append(f"{'✅' if total_words >= 1500 else '⚠️' if total_words >= 800 else '❌'} Content length: {total_words} words (aim for 1500+)")
            
            # Paragraph count check (15 points)
            valid_paragraphs = [p for p in soup.find_all('p') if p.get_text().strip()]
            if len(valid_paragraphs) >= 10:
                score += 15
            elif len(valid_paragraphs) >= 5:
                score += 8
            reasons.append(f"{'✅' if len(valid_paragraphs) >= 10 else '⚠️' if len(valid_paragraphs) >= 5 else '❌'} Paragraph count: {len(valid_paragraphs)} (aim for 10+)")

            # NEW CHECKS (Hidden point deductions)
            
            # Multiple H1 check
            h1_tags = soup.find_all('h1')
            if len(h1_tags) > 1:
                hidden_deductions += 5
                reasons.append("❌ Multiple H1 tags detected - use only one H1 per page")
            
            # Missing H1 check
            if len(h1_tags) == 0:
                hidden_deductions += 5
                reasons.append("❌ Missing H1 tag - add a main heading")
            
            # Missing H2 check
            h2_tags = soup.find_all('h2')
            if len(h2_tags) == 0:
                hidden_deductions += 5
                reasons.append("❌ Missing H2 tags - add subheadings")
            
            # Missing H3 check
            h3_tags = soup.find_all('h3')
            if len(h3_tags) == 0:
                hidden_deductions += 5
                reasons.append("❌ Missing H3 tags - add sub-subheadings")
            
            # Internal links check
            internal_links = soup.find_all('a', href=lambda href: href and not href.startswith(('http', 'https', '//')))
            if len(internal_links) == 0:
                hidden_deductions += 5
                reasons.append("❌ No internal links found - add relevant internal links")
            
            # Spelling check
            words = set(nltk.word_tokenize(text))
            english_words = set(nltk.corpus.words.words())
            misspelled = [word for word in words if word.isalpha() and word.lower() not in english_words]
            if misspelled:
                hidden_deductions += 5
                reasons.append("❌ Spelling errors detected - proofread content")
            
            # Grammar check
            grammar_matches = language_tool.check(text)
            if grammar_matches:
                hidden_deductions += 5
                reasons.append("❌ Grammar errors detected - proofread content")
            
            # Apply hidden deductions to final score
            final_score = max(0, score - hidden_deductions)
            
            return {
                'score': min(final_score, 100),
                'reasons': reasons
            }
            
        except Exception as e:
            print(f"Score content error: {str(e)}")
            return {
                'score': 0,
                'reasons': [f"❌ Error in score calculation: {str(e)}"]
            }

    def _calculate_keyword_density(self, text, keyword):
        """Calculate keyword density percentage"""
        words = text.lower().split()
        total_words = len(words)
        if total_words == 0:
            return 0
        
        keyword_count = 0
        keyword_words = keyword.lower().split()
        
        # Count occurrences using sliding window
        for i in range(len(words) - len(keyword_words) + 1):
            window = ' '.join(words[i:i + len(keyword_words)])
            if self._check_keyword_match(window, keyword):
                keyword_count += 1
        
        return (keyword_count * len(keyword_words) / total_words) * 100

    def _check_fluff_content(self, text):
        """Check for fluff content and return a score (higher is worse)"""
        fluff_phrases = [
            'very', 'really', 'just', 'quite', 'basically', 'actually',
            'definitely', 'literally', 'obviously', 'simply',
            'world-class', 'best-in-class', 'cutting-edge', 'next-generation',
            'industry-leading', 'revolutionary', 'game-changing'
        ]
        
        text_lower = text.lower()
        fluff_score = sum(text_lower.count(phrase) for phrase in fluff_phrases)
        return fluff_score

    def _check_hidden_content(self, soup):
        """Check for potentially hidden content"""
        hidden_indicators = [
            {'style': 'display: none'},
            {'style': 'visibility: hidden'},
            {'class': 'hidden'},
            {'style': 'text-indent: -9999px'},
            {'style': 'font-size: 0'},
            {'style': 'opacity: 0'}
        ]
        
        for indicator in hidden_indicators:
            if soup.find(attrs=indicator):
                return True
        return False

    def _check_keyword_match(self, text, keyword):
        """Enhanced keyword matching that handles variations"""
        text = text.lower()
        keyword = keyword.lower()
        
        # Direct match
        if keyword in text:
            return True
            
        # Split into words for more flexible matching
        keyword_words = set(keyword.split())
        text_words = text.split()
        
        # Check for all words in any order within a reasonable window
        window_size = len(keyword_words) + 2
        for i in range(len(text_words) - window_size + 1):
            window = set(text_words[i:i + window_size])
            if keyword_words.issubset(window):
                return True
                
        # Check for hyphenated variations
        text_without_hyphens = text.replace('-', ' ')
        if keyword in text_without_hyphens:
            return True
            
        # Check for concatenated variations
        text_concat = ''.join(text.split())
        keyword_concat = ''.join(keyword.split())
        if keyword_concat in text_concat:
            return True
            
        return False

    def score(self, html, tfidf, lsi, keyword_list):
        """Endpoint for real-time scoring"""
        try:
            # Ensure we have valid HTML
            if not html or not isinstance(html, str):
                return [0, ["❌ Invalid content"]]

            # Clean up the HTML if it's just the editor content
            if not html.startswith('<html>'):
                html = f"<html><body>{html}</body></html>"

            # Get the score data
            score_data = self.score_content(html, keyword_list)
            
            # Add LSI keyword scoring
            if lsi and isinstance(lsi, list) and len(lsi) > 0:
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text().lower()
                lsi_found = 0
                
                # Count how many LSI keywords are used
                for keyword, _ in lsi:
                    if self._check_keyword_match(text, keyword):
                        lsi_found += 1
                
                # Add LSI scoring to the results
                if lsi_found > 0:
                    lsi_score = min(lsi_found * 2, 10)  # Max 10 points for LSI
                    score_data['score'] += lsi_score
                    score_data['reasons'].append(f"✅ Found {lsi_found} LSI keywords")
                else:
                    score_data['reasons'].append("⚠️ No LSI keywords found")

            # Ensure score stays within bounds
            score_data['score'] = min(max(score_data['score'], 0), 100)
            
            return [score_data['score'], score_data['reasons']]
            
        except Exception as e:
            print(f"Scoring error: {str(e)}")  # Add debugging
            return [0, [f"❌ Error calculating score: {str(e)}"]]

    def get_max_tokens(self, model):
        model_limits = {
            'gpt-4': 8192,
            'gpt-4-32k': 32768,
            'gpt-3.5-turbo': 4096,
            'gpt-3.5-turbo-16k': 16384
        }
        return model_limits.get(model, 4096)

    def calculate_prompt_tokens(self, prompt):
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(prompt) // 4
    
    def generate_content(self, params):
        try:
            keyword = params['keyword']
            url = params.get('url', '')
            settings = params.get('settings', {})
            word_target = settings.get('words', 2950)
            headings = settings.get('headings', 9)
            paragraphs = settings.get('paragraphs', 13)
            images = settings.get('images', 8)
            model = settings.get('model', 'gpt-4')

            # Extract brand name if URL is provided
            brand_name = self.extract_brand_name(url) if url else "our company"

            # Generate image prompts first
            image_prompts = []
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional image description writer."},
                        {"role": "user", "content": f"Create {images} professional, commercial image descriptions for DALL-E about '{keyword}'. Focus on high-quality, business-appropriate imagery. Keep each description under 100 characters."}
                    ]
                )
                image_prompts = [desc.strip() for desc in response.choices[0].message.content.split('\n') if desc.strip()][:images]
            except Exception as e:
                print(f"Error generating image prompts: {str(e)}")
                image_prompts = [f"Professional {keyword} {i+1}" for i in range(images)]

            # Generate images using DALL-E
            image_urls = []
            for prompt in image_prompts:
                try:
                    response = self.openai_client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1
                    )
                    image_urls.append(response.data[0].url)
                except Exception as e:
                    print(f"Error generating image: {str(e)}")

            # Calculate sections for long content
            section_length = 2000  # Max tokens per section
            num_sections = max(1, (word_target + section_length - 1) // section_length)
            words_per_section = word_target // num_sections
            headings_per_section = max(2, headings // num_sections)
            paragraphs_per_section = max(3, paragraphs // num_sections)

            # Generate outline first
            outline_prompt = f"""Create a detailed {headings}-section outline for a commercial service page about '{keyword}'.
Focus on commercial intent, benefits, and service features.
Include H1, H2, and H3 headings."""

            outline_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an SEO content strategist specializing in commercial service pages."},
                    {"role": "user", "content": outline_prompt}
                ]
            )
            outline = outline_response.choices[0].message.content.split('\n')

            # Generate content in sections
            all_content = []
            for i in range(num_sections):
                is_first = i == 0
                is_last = i == len(outline) - 1
                section_outline = outline[i * headings_per_section:(i + 1) * headings_per_section]
                
                prompt = f"""Generate section {i+1}/{num_sections} of a commercial service page about '{keyword}'.
Use {brand_name} as the brand name.

Requirements:
- Target {words_per_section} words
- Follow this section outline:
{chr(10).join(section_outline)}
- Create {paragraphs_per_section} detailed paragraphs
- Use commercial intent and persuasive language
- Maintain natural keyword density (1-3%)
- Include trust indicators and social proof
- Add clear calls-to-action
{'- Start with main H1 title and introduction' if is_first else ''}
{'- End with strong call-to-action and contact section' if is_last else ''}

Format with proper HTML tags (<h1>, <h2>, <h3>, <p>). DO NOT use quotation marks in headings."""

                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional SEO copywriter specializing in commercial service pages."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )

                section_content = response.choices[0].message.content
                all_content.append(section_content)

            # Combine content and insert images
            complete_content = "\n\n".join(all_content)
            
            # Insert images at strategic positions
            if image_urls:
                paragraphs = complete_content.split('\n\n')
                spacing = max(1, len(paragraphs) // (len(image_urls) + 1))
                
                for i, url in enumerate(image_urls):
                    pos = (i + 1) * spacing
                    if pos < len(paragraphs):
                        img_html = f'<img src="{url}" alt="{image_prompts[i]}" style="width:100%; max-width:800px; height:auto; margin:20px 0;">'
                        paragraphs.insert(pos, img_html)
                
                complete_content = '\n\n'.join(paragraphs)

            return complete_content

        except Exception as e:
            print(f"Content generation error: {str(e)}")
            return f"<p>Error generating content: {str(e)}</p>"

    def generate_long_content(self, params):
        """Generate long-form content by splitting into sections"""
        try:
            keyword = params['keyword']
            settings = params.get('settings', {})
            total_words = settings.get('words', 2950)
            total_headings = settings.get('headings', 9)
            total_paragraphs = settings.get('paragraphs', 13)
            total_images = settings.get('images', 8)
            model = settings.get('model', 'gpt-4-32k')  # Default to 32k for long content

            # Calculate sections
            num_sections = (total_words + 1999) // 2000  # Round up division
            words_per_section = total_words // num_sections
            headings_per_section = max(2, total_headings // num_sections)
            paragraphs_per_section = max(3, total_paragraphs // num_sections)
            images_per_section = max(1, total_images // num_sections)

            all_content = []
            outline = self.generate_content_outline(keyword, num_sections)

            for i, section_topic in enumerate(outline):
                is_first = i == 0
                is_last = i == len(outline) - 1
                
                # Adjust last section to meet total word count
                if is_last:
                    remaining_words = total_words - (words_per_section * (num_sections - 1))
                    words_per_section = remaining_words

                prompt = f"""Generate section {i+1}/{num_sections} of a commercial service page about '{keyword}'.
This section should focus on: {section_topic}

Requirements:
- Word count: {words_per_section} words
- Include {headings_per_section} headings {'(including H1 for main title)' if is_first else '(H2-H3 only)'}
- Create {paragraphs_per_section} paragraphs
- Leave space for {images_per_section} images with [IMAGE] placeholders
- Use commercial intent and persuasive language
{'- Start with main H1 title and introduction' if is_first else ''}
{'- End with strong call-to-action and contact section' if is_last else ''}

Structure with proper HTML tags (<h1>, <h2>, <h3>, <p>)."""

                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional SEO content writer specializing in commercial service pages."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )

                section_content = response.choices[0].message.content
                all_content.append(section_content)

            # Combine all sections
            complete_content = "\n\n".join(all_content)
            return self.insert_image_placeholders(complete_content, total_images)

        except Exception as e:
            return f"<p>Error generating long-form content: {str(e)}</p>"

    def generate_content_outline(self, keyword, num_sections):
        """Generate an outline for long-form content"""
        prompt = f"""Create a {num_sections}-section outline for a commercial service page about '{keyword}'.
Each section should cover a unique aspect of the topic.
Format: Return only the section topics, one per line."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use 3.5 for outline generation to save tokens
                messages=[
                    {"role": "system", "content": "You are an SEO content strategist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            outline = response.choices[0].message.content.strip().split('\n')
            return outline[:num_sections]  # Ensure we have exactly the number of sections needed
        except Exception:
            # Fallback outline if API call fails
            return [
                f"Introduction to {keyword}",
                "Benefits and Features",
                "Services Overview",
                "Why Choose Us",
                "Pricing and Packages",
                "Customer Testimonials",
                "Contact and Next Steps"
            ][:num_sections]

    def insert_image_placeholders(self, content, num_images):
        image_prompts = self.generate_image_prompts(content, num_images)
        
        # Insert image placeholders at appropriate positions
        paragraphs = content.split('\n\n')
        spacing = max(1, len(paragraphs) // (num_images + 1))
        
        for i, prompt in enumerate(image_prompts):
            pos = (i + 1) * spacing
            if pos < len(paragraphs):
                paragraphs.insert(pos, f'\n<div class="image-placeholder" data-prompt="{prompt}">[IMAGE: {prompt}]</div>\n')
        
        return '\n\n'.join(paragraphs)

    def generate_image_prompts(self, content, num_images):
        try:
            prompt = f"""Based on the following content, generate {num_images} professional image descriptions for DALL-E.
Each description should be clear, specific, and focus on high-quality commercial imagery.
Keep each description under 100 characters.

Content:
{content}"""

            response = self.openai_client.chat.completions.create(
                model='gpt-3.5-turbo',  # Using 3.5 for image prompts to save tokens
                messages=[
                    {"role": "system", "content": "You are a professional image description writer for commercial websites."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            prompts = response.choices[0].message.content.strip().split('\n')
            return [p.strip() for p in prompts if p.strip()][:num_images]

        except Exception as e:
            print(f"Error generating image prompts: {str(e)}")
            return [f"Professional {i+1}" for i in range(num_images)]

    def generate_content_in_parts(self, base_prompt, model, word_target, image_prompts):
        """Generate content in multiple parts for longer content"""
        try:
            # First part: Introduction and first half
            first_prompt = f"{base_prompt}\nGenerate the first {word_target//2} words with introduction and first sections."
            response1 = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert conversion copywriter for commercial service pages."},
                    {"role": "user", "content": first_prompt}
                ],
                max_tokens=self.get_completion_tokens(model, word_target//2)
            )

            # Second part: Remaining content
            second_prompt = (
                f"{base_prompt}\n"
                f"Continue from this content and complete the page:\n\n{response1.choices[0].message.content}"
            )
            response2 = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert conversion copywriter for commercial service pages."},
                    {"role": "user", "content": second_prompt}
                ],
                max_tokens=self.get_completion_tokens(model, word_target//2)
            )

            # Combine content
            content = response1.choices[0].message.content + response2.choices[0].message.content

            # Insert image placeholders
            if image_prompts:
                content = self.insert_image_placeholders(content, image_prompts)

            return content
        except Exception as e:
            return f"<p><b>Error generating content in parts:</b> {str(e)}</p>"

    def get_completion_tokens(self, model, word_target):
        """Calculate appropriate completion tokens based on model and word target"""
        # Approximate tokens needed (roughly 1.5 tokens per word)
        needed_tokens = int(word_target * 1.5)
        
        model_limits = {
            "gpt-4": 7000,        # Leave some room for prompt
            "gpt-3.5-turbo-16k": 15000,
            "gpt-4-32k": 31000,
            "gpt-3.5-turbo": 3000
        }
        
        max_tokens = model_limits.get(model, 3000)
        return min(needed_tokens, max_tokens)

    def regenerate_content(self, original_prompt, model, target_words, actual_words, images):
        """Regenerate content if it's too short"""
        try:
            # Split content generation into smaller chunks
            parts = []
            words_per_part = min(2000, target_words // 2)  # Generate in 2000-word chunks or smaller
            remaining_words = target_words
            
            while remaining_words > 0:
                current_target = min(words_per_part, remaining_words)
                part_prompt = f"""Generate part of the content with exactly {current_target} words.
                Previous parts generated: {len(parts)}
                {original_prompt}"""
                
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert SEO content writer. Generate exactly the requested word count."},
                        {"role": "user", "content": part_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=self.get_max_tokens(model),
                    timeout=60
                )
                
                if response and response.choices:
                    parts.append(response.choices[0].message.content)
                    remaining_words -= current_target
                else:
                    raise Exception("No content generated in regeneration attempt")
            
            return "\n\n".join(parts)
            
        except Exception as e:
            return f"<p>Error regenerating content: {str(e)}</p>"

    def analyse(self, params):
        try:
            # Input validation
            if not params.get('keyword'):
                return json.dumps({
                    "error": "Please enter a primary keyword"
                })
            
            keyword = params['keyword']
            url = params.get('url', '')
            country = params.get('country', 'Australia')
            secondary_keywords = params.get('secondary_keywords', [])
            keyword_list = [keyword] + secondary_keywords

            # Set timeout for requests
            timeout = 30  # 30 seconds timeout

            # Get page HTML with timeout
            if url:
                try:
                    html = self.extract_content(url)
                    if not html:
                        return json.dumps({
                            "error": "Could not fetch content from the URL. Please check if the URL is accessible."
                        })
                except Exception as e:
                    return json.dumps({
                        "error": f"Error accessing URL: {str(e)}"
                    })
            else:
                html = "<html><body></body></html>"  # Empty page if no URL

            # Basic content analysis
            try:
                stats = self.analyse_content(html)
                brand_name = self.extract_brand_name(url) if url else None
            except Exception as e:
                return json.dumps({
                    "error": f"Error analyzing content: {str(e)}"
                })

            # Competitor analysis with timeout protection
            try:
                links = self.get_google_results(keyword, country)
                comp_htmls, comp_stats = [], []

                for link in links[:5]:  # Limit to top 5 competitors
                    if link == url:
                        continue
                    try:
                        html_comp = self.extract_content(link)
                        if len(html_comp) > 100:
                            domain = urlparse(link).netloc.replace("www.", "")
                            comp_stats.append({
                                "url": link,
                                "domain": domain,
                                **self.analyse_content(html_comp)
                            })
                            comp_htmls.append(BeautifulSoup(html_comp, 'html.parser').get_text())
                    except Exception:
                        continue  # Skip competitor if there's an error
            except Exception as e:
                # Continue with empty competitor data if Google API fails
                links, comp_htmls, comp_stats = [], [], []

            # Keyword analysis
            try:
                base_text = BeautifulSoup(html, 'html.parser').get_text().lower()
                tfidf = self.tfidf_keywords(base_text, comp_htmls) if comp_htmls else []
                lsi = self.extract_lsi_keywords(base_text)
            except Exception as e:
                return json.dumps({
                    "error": f"Error analyzing keywords: {str(e)}"
                })

            # Score calculation
            try:
                keyword_analysis = self.analyze_keyword_presence(html, keyword_list)
                score_data = self.score_content(html, keyword_list)
                score = score_data['score']
                reasons = score_data['reasons']
            except Exception as e:
                return json.dumps({
                    "error": f"Error calculating score: {str(e)}"
                })

            # Return complete analysis
            return json.dumps({
                "user_url": url,
                "keyword": keyword,
                "country": country,
                "brand_name": brand_name,
                "user_stats": stats,
                "competitors": comp_stats,
                "tfidf_keywords": tfidf,
                "lsi_keywords": lsi,
                "keyword_analysis": keyword_analysis,
                "score": score,
                "score_reasons": reasons
            })

        except Exception as e:
            return json.dumps({
                "error": f"Analysis failed: {str(e)}"
            })

    def analyze_keyword_presence(self, html, keywords):
        if not keywords:
            return {
                'title': False,
                'h1': False,
                'meta_desc': False,
                'first_para': False,
                'headings': [],
                'density': 0,
                'count': 0
            }
        
        soup = BeautifulSoup(html, 'html.parser')
        results = {
            'title': False,
            'h1': False,
            'meta_desc': False,
            'first_para': False,
            'headings': [],
            'density': 0,
            'count': 0
        }
        
        try:
            # Get text content for density calculation
            text = soup.get_text().lower()
            total_words = len(text.split())
            
            # Get first non-empty paragraph
            first_para = next((p for p in soup.find_all('p') if p.get_text().strip()), None)
            if first_para:
                first_para_text = first_para.get_text().lower()
                results['first_para'] = any(self._check_keyword_match(first_para_text, keyword.lower()) for keyword in keywords)
            
            # Check title
            title = soup.find('title')
            if title:
                results['title'] = any(self._check_keyword_match(title.text.lower(), keyword.lower()) for keyword in keywords)
            
            # Check H1
            h1s = soup.find_all('h1')
            for h1 in h1s:
                if any(self._check_keyword_match(h1.text.lower(), keyword.lower()) for keyword in keywords):
                    results['h1'] = True
                    break
            
            # Check meta description
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                if any(self._check_keyword_match(meta_desc.get('content').lower(), keyword.lower()) for keyword in keywords):
                    results['meta_desc'] = True
            
            # Check all headings
            for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if any(self._check_keyword_match(h.text.lower(), keyword.lower()) for keyword in keywords):
                    results['headings'].append(h.name)
            
            # Calculate keyword density
            main_keyword = keywords[0].lower()
            keyword_count = 0
            words = text.split()
            
            for i in range(len(words)):
                window = ' '.join(words[i:i + len(main_keyword.split())])
                if self._check_keyword_match(window, main_keyword):
                    keyword_count += 1
            
            results['count'] = keyword_count
            if total_words > 0:
                results['density'] = (keyword_count / total_words) * 100
            
            return results
        except Exception as e:
            print(f"Keyword presence analysis error: {str(e)}")
            return results

    def generate_headline_suggestions(self, keyword):
        """Generate commercial service page headlines"""
        location = keyword.split()[-1] if len(keyword.split()) > 1 else "your area"
        
        prompt = f"""Generate 9 commercial service page headlines for '{keyword}'. Include:
- Location-based headlines using '{location}'
- Trust and authority signals
- Clear value propositions
- Urgent need or problem-solving focus
- Commercial intent (not blog-style)

Examples:
- Trusted [Service] Experts in {location} | 24/7 Service
- Need a Professional [Service] in {location}? Same-Day Service
- {location}'s Most Reliable [Service] Solutions | Free Quote
- Expert [Service] Services in {location} | Licensed & Insured

Format as a list. Focus on commercial service pages, not blog posts."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in writing commercial service page headlines that convert."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content
            return [line.strip().replace('- ', '') for line in text.split('\n') if line.strip()]
        except Exception as e:
            return [f"Error generating headlines: {str(e)}"]

    def build_topic_clusters(self, keyword):
        prompt = (
            f"Create a comprehensive topic cluster for '{keyword}'. Include:\n"
            "- Main pillar topic\n"
            "- 5 subtopics with clear SEO value\n"
            "- 3 long-tail variations\n"
            "Format as bullet points."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in SEO topic clustering and content architecture."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content
            return [line.strip().replace('- ', '') for line in text.split('\n') if line.strip()]
        except Exception as e:
            return [f"Error generating clusters: {str(e)}"]

    def validate_outline(self, headings):
        outline_text = "\n".join(headings)
        prompt = (
            f"Analyze this content outline for SEO effectiveness:\n{outline_text}\n\n"
            "Consider:\n"
            "- Heading hierarchy\n"
            "- Topic coverage\n"
            "- User intent match\n"
            "- Content flow\n"
            "Provide specific improvement suggestions."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in SEO content structure and outline optimization."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error validating outline: {str(e)}"

    def rewrite_paragraph(self, paragraph, keyword):
        prompt = (
            f"Rewrite the following paragraph to improve clarity, SEO, and engagement while keeping the keyword '{keyword}':\n\n"
            f"{paragraph}"
        )
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error rewriting paragraph: {str(e)}"

    def detect_fluff(self, content):
        fluff_phrases = [
            "our team of experts", "best in the industry", "we pride ourselves",
            "world-class service", "leading provider", "highly experienced team",
            "tailored solutions", "bespoke services", "cutting-edge technology",
            "state-of-the-art", "industry-leading", "next-generation",
            "revolutionary", "game-changing", "innovative solution",
            "unparalleled service", "exceeding expectations", "second to none"
        ]
        
        findings = []
        content_lower = content.lower()
        
        for phrase in fluff_phrases:
            if phrase.lower() in content_lower:
                start = content_lower.index(phrase.lower())
                # Get surrounding context
                context_start = max(0, start - 50)
                context_end = min(len(content), start + len(phrase) + 50)
                context = content[context_start:context_end]
                findings.append({
                    'phrase': phrase,
                    'context': f"...{context}...",
                    'suggestion': self.get_fluff_suggestion(phrase)
                })
        
        return findings

    def get_fluff_suggestion(self, phrase):
        suggestions = {
            "our team of experts": "Specify expertise: '15+ years in cybersecurity' or 'certified AWS architects'",
            "best in the industry": "Use specific metrics: '98% customer satisfaction' or '50% faster delivery'",
            "we pride ourselves": "Show, don't tell: 'Maintained 99.9% uptime' or 'Completed 1000+ projects'",
            "world-class service": "Be specific: 'Same-day response time' or '24/7 support available'",
            "leading provider": "Use numbers: 'Serving 10,000+ clients' or 'Operating in 15 countries'",
            "highly experienced team": "Quantify experience: 'Combined 50 years experience' or 'Certified by X'",
            "tailored solutions": "Explain how: 'Customized based on your industry requirements' or 'Personalized after thorough analysis'",
            "bespoke services": "Be specific: 'Custom-built for your exact specifications' or 'Individually designed workflows'"
        }
        return suggestions.get(phrase, "Replace with specific, measurable achievements or capabilities")

    def compression_score(self, content):
        """
        Calculate various readability and compression metrics
        """
        words = content.split()
        word_count = len(words)
        char_count = len(content.replace(' ', ''))
        sentences = content.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        if word_count == 0 or sentence_count == 0:
            return {
                'compression_ratio': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'suggestions': ['Text is too short for analysis']
            }
            
        metrics = {
            'compression_ratio': round(char_count / word_count, 2),
            'avg_word_length': round(char_count / word_count, 2),
            'avg_sentence_length': round(word_count / sentence_count, 2),
            'suggestions': []
        }
        
        # Add suggestions based on metrics
        if metrics['avg_sentence_length'] > 25:
            metrics['suggestions'].append('Consider breaking down longer sentences for better readability')
        if metrics['avg_word_length'] > 6:
            metrics['suggestions'].append('Try using simpler words where possible')
        if metrics['compression_ratio'] > 5:
            metrics['suggestions'].append('Content might be too dense - consider adding more whitespace and shorter paragraphs')
            
        return metrics

api = Api()
webview.create_window("SEO Analyzer", "frontend/index.html", js_api=api, width=1400, height=800)
webview.start()
