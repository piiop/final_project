# %%

from typing import Dict, List, Optional
import json
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import numpy as np

import tensorflow as tf
tf.random.set_seed(42)
import numpy as np
np.random.seed(42)



# %%


class ArtAnalyzer:
    def __init__(self, cnn_model_path: str, style_labels: List[str], llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        self.style_labels = style_labels
        
        # Load LLM and tokenizer
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        # Pre-define question type keyword sets for faster matching
        self.technique_keywords = {"technique", "style", "painted", "created", "made"}
        self.historical_keywords = {"history", "period", "when", "era"}
        self.influence_keywords = {"influence", "impact", "change", "affect"}
        
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
            print("Model summary:")
            model.summary()
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def analyze_image(self, image) -> Dict[str, float]:
        try:
            if len(image.shape) == 3:
                image = tf.expand_dims(image, 0)
            
            # Convert to numpy for the debug prints
            image_np = image.numpy()
            print("Image shape before prediction:", image_np.shape)
            print("Image value range:", image_np.min(), "to", image_np.max())
                    
            # Get predictions
            probabilities = self.cnn_model.predict(image, verbose=0)[0]
            
            
            print("\nDetailed prediction mapping:")
            for i, (style, prob) in enumerate(zip(self.style_labels, probabilities)):
                print(f"Index {i:2d}: {style:20s} = {prob:.4f}")
            
            # Create dictionary of style predictions
            result = {
                self.style_labels[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            return result
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            raise Exception(f"Error during image analysis: {str(e)}")
        
    def clean_response(response: str) -> str:
        """Clean up LLM response by removing prompt artifacts and extra whitespace"""
        
        # Split on known markers and take the relevant part
        if "Answer:" in response:
            response = response.split("Answer:", 1)[1]
        
        # Remove any trailing prompt parts
        markers_to_remove = [
            "Question:",
            "Context about the artwork:",
            "Please provide a detailed",
            "Focus on being accurate"
        ]
        for marker in markers_to_remove:
            if marker in response:
                response = response.split(marker, 1)[0]
        
        # Clean up whitespace
        response = response.strip()
        response = ' '.join(response.split())  # Replace multiple spaces with single space
        
        return response    
            
    def generate_response_with_context(self, question: str, style_predictions: Dict[str, float]) -> Tuple[str, str]:
        """Combined question classification and context generation"""
        question_lower = question.lower()
        
        # Determine type and get top styles in one pass
        top_styles = sorted(style_predictions.items(), key=lambda x: x[1], reverse=True)[:2]
        primary_style = top_styles[0][0].lower().replace(" ", "_")
        style_info = self.style_info.get(primary_style, {})
        
        # Determine context type using sets
        if any(keyword in question_lower for keyword in self.technique_keywords):
            question_type = "technique"
        elif any(keyword in question_lower for keyword in self.historical_keywords):
            question_type = "historical"
        elif any(keyword in question_lower for keyword in self.influence_keywords):
            question_type = "influence"
        else:
            question_type = "general"
            
        # Use your existing context templates
        context_templates = {
            "technique": f"""
                The artwork shows characteristics of {primary_style} ({top_styles[0][1]:.1%} confidence).
                Key techniques in {primary_style} include: {', '.join(style_info.get('techniques', []))}
                Common characteristics: {', '.join(style_info.get('characteristics', []))}
            """,
            "historical": f"""
                This style emerged in {style_info.get('period', 'unknown period')}.
                Historical context: {style_info.get('historical_context', '')}
                Notable artists: {', '.join(style_info.get('notable_artists', []))}
                Key works: {', '.join(style_info.get('key_works', []))}
            """,
            "influence": f"""
                {style_info.get('influence', '')}
                Impact on art history: {style_info.get('impact', '')}
                This style influenced: {', '.join(style_info.get('influenced', []))}
            """,
            "general": f"""
                The artwork primarily shows {primary_style} characteristics ({top_styles[0][1]:.1%} confidence).
                Secondary style influence: {top_styles[1][0]} ({top_styles[1][1]:.1%} confidence).
                Period: {style_info.get('period', '')}
                Key characteristics: {', '.join(style_info.get('characteristics', []))}
            """
        }
        
        return context_templates[question_type], question_type

    def answer_question(self, question: str, style_predictions: Dict[str, float]) -> str:
        """Generate a response using TinyLlama with optimized parameters"""
        context, _ = self.generate_response_with_context(question, style_predictions)
        
        prompt = f"""
        Context about the artwork: {context}
        
        Question: {question}
        
        Please provide a detailed but concise answer based on the context provided.
        Focus on being accurate and informative while maintaining a conversational tone. Keep your response
        to less than 400 characters.
        
        Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.llm.generate(
            **inputs,
            max_length=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return self.clean_response(response)


