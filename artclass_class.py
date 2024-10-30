# %%

from typing import Dict, List, Optional
import json
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import numpy as np
import torch 

tf.random.set_seed(42)
np.random.seed(42)



# %%


class ArtAnalyzer:
    def __init__(self, cnn_model_path: str, style_labels: List[str]):
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        self.style_labels = style_labels
        
        # TinyLlama initialization
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load model 
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32  
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set model to eval mode for inference
        self.llm.eval()
        
        # Define keywords for question classification
        self.technique_keywords = {"technique", "how", "method", "painted", "style", "characteristics"}
        self.historical_keywords = {"when", "period", "history", "time", "era"}
        self.influence_keywords = {"influence", "impact", "affect", "inspire"}
        
        self.style_info = {
            "naive_art": {
                "period": "19th-20th century",
                "characteristics": [
                    "Childlike simplicity",
                    "Bright colors",
                    "Lack of formal perspective",
                    "Intuitive approach to composition"
                ],
                "notable_artists": ["Henri Rousseau", "Grandma Moses", "Niko Pirosmani"],
                "key_works": ["The Sleeping Gypsy", "The Dream"]
            },
            "baroque": {
                "period": "17th century",
                "characteristics": [
                    "Dramatic light and shadow",
                    "Dynamic composition",
                    "Emotional intensity",
                    "Rich detail and ornamentation"
                ],
                "notable_artists": ["Caravaggio", "Rembrandt", "Peter Paul Rubens"],
                "key_works": ["The Night Watch", "The Conversion of Saint Paul"]
            },
            "rococo": {
                "period": "18th century",
                "characteristics": [
                    "Ornate decoration",
                    "Light colors",
                    "Asymmetrical designs",
                    "Playful themes"
                ],
                "notable_artists": ["Jean-Honoré Fragonard", "François Boucher", "Antoine Watteau"],
                "key_works": ["The Swing", "Pilgrimage to Cythera"]
            },
            "romanticism": {
                "period": "Late 18th-mid 19th century",
                "characteristics": [
                    "Emphasis on emotion and nature",
                    "Dramatic landscapes",
                    "Historical subjects",
                    "Individual expression"
                ],
                "notable_artists": ["Eugène Delacroix", "J.M.W. Turner", "Caspar David Friedrich"],
                "key_works": ["Liberty Leading the People", "The Wanderer above the Sea of Fog"]
            },
            "art_deco": {
                "period": "1920s-1930s",
                "characteristics": [
                    "Geometric patterns",
                    "Luxurious materials",
                    "Streamlined forms",
                    "Bold colors and symmetry"
                ],
                "notable_artists": ["Tamara de Lempicka", "René Lalique", "Erté"],
                "key_works": ["Young Lady with Gloves", "The Wisdom of Athena"]
            },
            "american_realism": {
                "period": "Mid 19th-early 20th century",
                "characteristics": [
                    "Depiction of everyday life",
                    "Urban scenes",
                    "Social commentary",
                    "Precise detail"
                ],
                "notable_artists": ["Edward Hopper", "Thomas Eakins", "Winslow Homer"],
                "key_works": ["Nighthawks", "The Gross Clinic"]
            },
            "art_nouveau": {
                "period": "1890-1910",
                "characteristics": [
                    "Organic, flowing lines",
                    "Nature-inspired forms",
                    "Decorative patterns",
                    "Integration of form and structure"
                ],
                "notable_artists": ["Alphonse Mucha", "Gustav Klimt", "Henri de Toulouse-Lautrec"],
                "key_works": ["The Four Seasons", "Gismonda"]
            },
            "expressionism": {
                "period": "Early 20th century",
                "characteristics": [
                    "Emotional intensity",
                    "Distorted forms",
                    "Bold colors",
                    "Subjective perspective"
                ],
                "notable_artists": ["Edvard Munch", "Ernst Ludwig Kirchner", "Emil Nolde"],
                "key_works": ["The Scream", "Berlin Street Scene"]
            },
            "modernism": {
                "period": "Late 19th-mid 20th century",
                "characteristics": [
                    "Abstract elements",
                    "Experimental approach",
                    "Rejection of tradition",
                    "Innovation focus"
                ],
                "notable_artists": ["Piet Mondrian", "Constantin Brancusi", "Georgia O'Keeffe"],
                "key_works": ["Broadway Boogie Woogie", "Bird in Space"]
            },
            "post_impressionism": {
                "period": "1886-1905",
                "characteristics": [
                    "Bold colors",
                    "Symbolic elements",
                    "Geometric forms",
                    "Emotional intensity"
                ],
                "notable_artists": ["Vincent van Gogh", "Paul Gauguin", "Paul Cézanne"],
                "key_works": ["The Starry Night", "The Card Players"]
            },
            "impressionism": {
                "period": "1860s-1880s",
                "characteristics": [
                    "Visible brushstrokes",
                    "Light and color emphasis",
                    "Outdoor painting",
                    "Capture of momentary effects"
                ],
                "notable_artists": ["Claude Monet", "Pierre-Auguste Renoir", "Edgar Degas"],
                "key_works": ["Impression, Sunrise", "Water Lilies"]
            },
            "high_renaissance": {
                "period": "1490-1527",
                "characteristics": [
                    "Perfect proportion",
                    "Mathematical precision",
                    "Idealized beauty",
                    "Sfumato technique"
                ],
                "notable_artists": ["Leonardo da Vinci", "Michelangelo", "Raphael"],
                "key_works": ["Mona Lisa", "The Last Supper"]
            },
            "cubism": {
                "period": "1907-1920s",
                "characteristics": [
                    "Geometric shapes",
                    "Multiple perspectives simultaneously",
                    "Fragmented forms",
                    "Monochromatic or limited color palette"
                ],
                "notable_artists": ["Pablo Picasso", "Georges Braque", "Juan Gris"],
                "key_works": ["Les Demoiselles d'Avignon", "Portrait of Daniel-Henry Kahnweiler"]
            },
            "surrealism": {
                "period": "1924-1950s",
                "characteristics": [
                    "Dream-like scenes",
                    "Juxtaposed elements",
                    "Unconscious imagery",
                    "Psychological themes"
                ],
                "notable_artists": ["Salvador Dalí", "René Magritte", "Max Ernst"],
                "key_works": ["The Persistence of Memory", "The Son of Man"]
            },
            "abstract_expressionism": {
                "period": "1940s-1950s",
                "characteristics": [
                    "Spontaneous, improvisational work",
                    "Large-scale canvases",
                    "Emotional intensity",
                    "Non-representational forms"
                ],
                "notable_artists": ["Jackson Pollock", "Willem de Kooning", "Mark Rothko"],
                "key_works": ["No. 5, 1948", "Woman I"]
            },
            "art_informel": {
                "period": "1940s-1950s",
                "characteristics": [
                    "Spontaneous gestural painting",
                    "Abstract compositions",
                    "Emphasis on material qualities",
                    "Rejection of geometric abstraction"
                ],
                "notable_artists": ["Jean Dubuffet", "Antoni Tàpies", "Jean Fautrier"],
                "key_works": ["Corps de dame", "Matter Painting"]
            },
            "mannerism": {
                "period": "16th century",
                "characteristics": [
                    "Elongated figures",
                    "Complex poses",
                    "Artificial colors",
                    "Technical sophistication"
                ],
                "notable_artists": ["Parmigianino", "El Greco", "Bronzino"],
                "key_works": ["Madonna with the Long Neck", "The Burial of Count Orgaz"]
            },
            "northern_renaissance": {
                "period": "15th-16th century",
                "characteristics": [
                    "Oil painting technique",
                    "Minute detail",
                    "Symbolic realism",
                    "Domestic scenes"
                ],
                "notable_artists": ["Jan van Eyck", "Albrecht Dürer", "Hieronymus Bosch"],
                "key_works": ["Arnolfini Portrait", "The Garden of Earthly Delights"]
            },
            "symbolism": {
                "period": "1886-1910",
                "characteristics": [
                    "Mystical themes",
                    "Personal mythology",
                    "Dream imagery",
                    "Spiritual content"
                ],
                "notable_artists": ["Gustav Klimt", "Odilon Redon", "Gustave Moreau"],
                "key_works": ["The Kiss", "The Cyclops"]
            },
            "early_renaissance": {
                "period": "14th-15th century",
                "characteristics": [
                    "Linear perspective",
                    "Naturalistic representation",
                    "Religious themes",
                    "Tempera painting"
                ],
                "notable_artists": ["Masaccio", "Fra Angelico", "Botticelli"],
                "key_works": ["The Tribute Money", "Primavera"]
            },
            "minimalism": {
                "period": "1960s-1970s",
                "characteristics": [
                    "Geometric abstraction",
                    "Simplicity of form",
                    "Industrial materials",
                    "Repetitive elements"
                ],
                "notable_artists": ["Donald Judd", "Dan Flavin", "Frank Stella"],
                "key_works": ["Untitled (Stack)", "The Marriage of Reason and Squalor, II"]
            },
            "neo_romantic": {
                "period": "1930s-1950s",
                "characteristics": [
                    "Emotional landscapes",
                    "Mystical elements",
                    "Nature focus",
                    "Poetic interpretation"
                ],
                "notable_artists": ["Graham Sutherland", "Paul Nash", "John Piper"],
                "key_works": ["Entrance to a Lane", "Totes Meer"]
            },
            "ukiyo_e": {
                "period": "17th-19th century",
                "characteristics": [
                    "Japanese woodblock prints",
                    "Flat perspective",
                    "Bold colors",
                    "Scenes of daily life and landscape"
                ],
                "notable_artists": ["Hokusai", "Hiroshige", "Utamaro"],
                "key_works": ["The Great Wave off Kanagawa", "53 Stations of the Tōkaidō"]
            },
            "pop_art": {
                "period": "1950s-1970s",
                "characteristics": [
                    "Use of commercial imagery",
                    "Bold colors",
                    "Mass culture references",
                    "Reproduction techniques"
                ],
                "notable_artists": ["Andy Warhol", "Roy Lichtenstein", "Claes Oldenburg"],
                "key_works": ["Campbell's Soup Cans", "Whaam!"]
            },
            "fauvism": {
                "period": "1904-1908",
                "characteristics": [
                    "Pure bright colors",
                    "Wild brushwork",
                    "Simplified forms",
                    "Emotional expression"
                ],
                "notable_artists": ["Henri Matisse", "André Derain", "Maurice de Vlaminck"],
                "key_works": ["The Green Stripe", "The Dance"]
            },
            "neoclassicism": {
                "period": "1750s-1850s",
                "characteristics": [
                    "Classical revival",
                    "Rational compositions",
                    "Historical themes",
                    "Clear drawing and modeling"
                ],
                "notable_artists": ["Jacques-Louis David", "Jean-Auguste-Dominique Ingres", "Antonio Canova"],
                "key_works": ["Oath of the Horatii", "Napoleon Crossing the Alps"]
            }
        }

    
    def _load_cnn_model(self, model_path: str = "art_classifier_curated.keras"):
        """
        Load the CNN model from a Keras save file
        
        Args:
            model_path: Path to the Keras model file, defaults to 'art_classifier_curated.keras'
        
        Returns:
            Loaded model ready for inference
        """
        try:
            from tensorflow import keras
            model = keras.models.load_model(model_path, compile=False)
            
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def analyze_image(self, image) -> Dict[str, float]:
        try:
            if len(image.shape) == 3:
                image = tf.expand_dims(image, 0)
            
            # Convert to numpy for the debug prints
            image_np = image.numpy()
            # print("Image shape before prediction:", image_np.shape)
            # print("Image value range:", image_np.min(), "to", image_np.max())
                    
            # Get predictions
            probabilities = self.cnn_model.predict(image, verbose=0)[0]
            
            
            # print("\nDetailed prediction mapping:")
            # for i, (style, prob) in enumerate(zip(self.style_labels, probabilities)):
            #     print(f"Index {i:2d}: {style:20s} = {prob:.4f}")
            
            # Create dictionary of style predictions
            result = {
                self.style_labels[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            return result
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            raise Exception(f"Error during image analysis: {str(e)}")
      
    def _clean_response(self, response: str, prompt: str, system_message: str) -> str:
        """Clean up the model response with enhanced filtering"""
        # Remove the prompt and system message from the response
        response = response.replace(prompt, "")
        response = response.replace(system_message, "")
        
        # Remove any system/user/assistant markers
        markers = ["<|system|>", "<|user|>", "<|assistant|>", "Rules:", "Complete Response:"]
        for marker in markers:
            response = response.replace(marker, "")
        
        # Remove style prediction prefixes and metadata
        prefixes_to_remove = [
            "Primary Predicted Style:",
            "Complete Response:",
            "Answer:",
            "Style:",
            "Primary Style:",
            "Here is a possible answer for this question:"
        ]
        for prefix in prefixes_to_remove:
            response = response.replace(prefix, "")
        
        # Remove cases where the model outputs the rules
        if "1." in response and "2." in response and "provide a complete response" in response.lower():
            response = response.split("1.")[0].strip()
        
        # Replace definitive artwork attributions with style-based language
        artwork_phrases = [
            "was painted by",
            "was created by",
            "is by",
            "was made by",
            "painted this",
            "created this"
        ]
        for phrase in artwork_phrases:
            if phrase in response.lower():
                response = response.replace(phrase, "appears to be in the style of")
        
        # Remove specific artwork titles
        famous_artworks = [
            "The Persistence of Memory",
            "The Starry Night",
            "Campbell's Soup Cans",
            "Whaam!",
            "Girl with a Pearl Earring"
        ]
        for artwork in famous_artworks:
            if artwork in response:
                response = response.split(artwork)[0].strip()
                response += " This work appears to be in the style of"
                # Get everything after the artwork title
                remaining = response.split(artwork)[1]
                # Keep any artist names that come after
                if "by" in remaining:
                    artist = remaining.split("by")[1].strip().split(".")[0]
                    response += f" {artist}."
                else:
                    response += "."
        
        # Clean up whitespace and formatting
        response = response.strip()
        response = " ".join(response.split())
        
        return response
            
    def generate_response_with_context(self, question: str, style_predictions: Dict[str, float]) -> Tuple[str, str]:
        """Generate context based on question type and style predictions"""
        question_lower = question.lower()
        
        # Get top styles
        top_styles = sorted(style_predictions.items(), key=lambda x: x[1], reverse=True)[:2]
        primary_style = top_styles[0][0].lower().replace(" ", "_")
        style_info = self.style_info.get(primary_style, {})
        
        # Determine question type
        if any(keyword in question_lower for keyword in self.technique_keywords):
            question_type = "technique"
        elif any(keyword in question_lower for keyword in self.historical_keywords):
            question_type = "historical"
        elif any(keyword in question_lower for keyword in self.influence_keywords):
            question_type = "influence"
        else:
            question_type = "general"
            
        # Format context based on question type
        context_templates = {
            "technique": f"""Style: {primary_style}
                Visual Elements: {', '.join(style_info.get('characteristics', []))}
                Cultural Impact: This style emerged in {style_info.get('period', '')} and transformed art by introducing new approaches to representation
                Key Examples: {', '.join(style_info.get('key_works', []))}""",
                
            "historical": f"""Period: {style_info.get('period', 'unknown period')}
                Key artists: {', '.join(style_info.get('notable_artists', []))}
                Development: Started in {style_info.get('period', '')}, focusing on {', '.join(style_info.get('characteristics', []))}
                Notable works: {', '.join(style_info.get('key_works', []))}""",
                
            "influence": f"""Primary style: {primary_style}
                Period: {style_info.get('period', '')}
                Major artists and contributions: {', '.join(style_info.get('notable_artists', []))}
                Cultural context: This movement focused on {', '.join(style_info.get('characteristics', []))}
                Key works: {', '.join(style_info.get('key_works', []))}""",
                
            "general": f"""Primary style: {primary_style} ({top_styles[0][1]:.1%} confidence)
                Period: {style_info.get('period', '')}
                Notable artists: {', '.join(style_info.get('notable_artists', []))}
                Characteristics: {', '.join(style_info.get('characteristics', []))}
                Key works: {', '.join(style_info.get('key_works', []))}"""
        }
    
        return context_templates[question_type], question_type

    def answer_question(self, question: str, style_predictions: Dict[str, float]) -> str:
        """Generate response using TinyLlama with separated system message"""
        # Get context and format prompt
        context, question_type = self.generate_response_with_context(question, style_predictions)
        
        # Separate system message
        system_message = """<|system|>
        You are an art history expert. Your primary role is to classify and explain artistic styles, not to identify specific artworks.
        Never make definitive claims about specific artworks or their creators.
        When discussing artwork, always use phrases like "appears to be in the style of" or "shows characteristics of".
        Focus on describing the style, its characteristics, and notable artists of the movement."""
        
        # User prompt with question-specific guidance
        user_prompt = f"""<|user|>
        When discussing this artwork:
        - Describe the style and its characteristics
        - Mention representative artists of this style
        - Avoid identifying specific artworks
        - Use "appears to be" rather than definitive statements
        
        Context: {context}
        Question: {question}

        <|assistant|>"""
        
        # Combine messages
        full_prompt = f"{system_message}\n{user_prompt}"
        
        # Debug: Print token count for monitoring
        token_count = len(self.tokenizer.encode(full_prompt))
        print(f"Prompt token count: {token_count}")
        
        # Tokenize with proper handling
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Generate response
        with torch.inference_mode():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.45,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and clean response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self._clean_response(response, user_prompt, system_message)
        
        return response


