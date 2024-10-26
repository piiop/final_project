# %%

from typing import Dict, List, Optional
import json
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%


class ArtAnalyzer:
    def __init__(self, cnn_model_path: str, style_labels: List[str], llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        self.style_labels = style_labels
        
        # Load LLM and tokenizer
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        self.style_info = {
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
            "conceptual_art": {
                "period": "1960s-present",
                "characteristics": [
                    "Emphasis on ideas over visual forms",
                    "Documentation and language-based work",
                    "Questioning of traditional art",
                    "Often installation-based"
                ],
                "notable_artists": ["Joseph Kosuth", "Sol LeWitt", "Marina Abramović"],
                "key_works": ["One and Three Chairs", "Wall Drawing Series"]
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
            "contemporary_realism": {
                "period": "1960s-present",
                "characteristics": [
                    "Photographic accuracy",
                    "Modern subjects",
                    "Technical precision",
                    "Everyday scenes"
                ],
                "notable_artists": ["Richard Estes", "Chuck Close", "Antonio López García"],
                "key_works": ["Telephone Booths", "Big Self-Portrait"]
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
            "contemporary": {
                "period": "1970s-present",
                "characteristics": [
                    "Diverse mediums and approaches",
                    "Global perspective",
                    "Social and political themes",
                    "Digital and new media"
                ],
                "notable_artists": ["Ai Weiwei", "Jeff Koons", "Damien Hirst"],
                "key_works": ["Sunflower Seeds", "Balloon Dog"]
            },
            "realism": {
                "period": "Mid-19th century",
                "characteristics": [
                    "Accurate depiction",
                    "Contemporary subjects",
                    "Social commentary",
                    "Unidealized scenes"
                ],
                "notable_artists": ["Gustave Courbet", "Jean-François Millet", "Honoré Daumier"],
                "key_works": ["The Stone Breakers", "The Gleaners"]
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
            "modern_art": {
                "period": "Late 19th-Mid 20th century",
                "characteristics": [
                    "Break from traditional techniques",
                    "Experimentation with form",
                    "New artistic perspectives",
                    "Emphasis on innovation"
                ],
                "notable_artists": ["Henri Matisse", "Marcel Duchamp", "Wassily Kandinsky"],
                "key_works": ["The Red Studio", "Fountain"]
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
            "late_renaissance": {
                "period": "Early-mid 16th century",
                "characteristics": [
                    "Increased complexity",
                    "Harmonious composition",
                    "Rich coloring",
                    "Advanced perspective"
                ],
                "notable_artists": ["Raphael", "Titian", "Tintoretto"],
                "key_works": ["School of Athens", "Assumption of the Virgin"]
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
            }
        }

    
    def _load_cnn_model(self, model_path: str = "art_classifier_best.keras"):
        """
        Load the CNN model from a Keras save file
        
        Args:
            model_path: Path to the Keras model file, defaults to 'art_classifier_best.keras'
        
        Returns:
            Loaded model ready for inference
        """
        try:
            from tensorflow import keras
            model = keras.models.load_model(model_path, compile=False)
            return model
        except FileNotFoundError:
            raise Exception(f"Model file not found at: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def analyze_image(self, image) -> Dict[str, float]:
        """
        Get style predictions from CNN model
        
        Args:
            image: Image tensor prepared for CNN (should be preprocessed)
                Expected shape: (1, height, width, channels)
        
        Returns:
            Dictionary mapping style names to prediction probabilities
        """
        try:
            # Add batch dimension if not present
            if len(image.shape) == 3:
                image = tf.expand_dims(image, 0)
                
            # Get predictions
            predictions = self.cnn_model.predict(image, verbose=0)
            
            # Apply softmax to get probabilities
            probabilities = tf.nn.softmax(predictions)[0]
            
            # Create dictionary of style predictions
            return {
                self.style_labels[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
        except Exception as e:
            raise Exception(f"Error during image analysis: {str(e)}")
        
    def get_style_information(self, style_name: str) -> Optional[Dict]:
        """Get detailed information about an art style"""
        # Convert style name to match dictionary keys
        style_key = style_name.lower().replace(" ", "_")
        return self.style_info.get(style_key)
    def generate_llm_context(self, style_predictions: Dict[str, float], question_type: str) -> str:
        """
        Generate relevant context for the LLM based on the question type and style predictions
        """
        # Get top 2 predicted styles
        top_styles = sorted(style_predictions.items(), key=lambda x: x[1], reverse=True)[:2]
        primary_style = top_styles[0][0].lower().replace(" ", "_")
        
        style_info = self.style_info.get(primary_style, {})
        
        # Create different context templates based on question type
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
        
        return context_templates.get(question_type, context_templates["general"])    
    def classify_question(self, question: str) -> str:
        """
        Determine the type of question being asked to generate appropriate context
        """
        question = question.lower()
        if any(word in question for word in ["technique", "style", "painted", "created", "made"]):
            return "technique"
        elif any(word in question for word in ["history", "period", "when", "era"]):
            return "historical"
        elif any(word in question for word in ["influence", "impact", "change", "affect"]):
            return "influence"
        return "general"

    def answer_question(self, question: str, style_predictions: Dict[str, float]) -> str:
        """
        Generate a response using Llama2 with appropriate context from style_info
        """
        # Determine question type
        question_type = self.classify_question(question)
        
        # Generate appropriate context
        context = self.generate_llm_context(style_predictions, question_type)
        
        # Create prompt for LLM
        prompt = f"""
        Context about the artwork: {context}
        
        Question: {question}
        
        Please provide a detailed but concise answer based on the context provided.
        Focus on being accurate and informative while maintaining a conversational tone.
        
        Answer:"""
        
        # Generate response using LLM
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.llm.generate(**inputs, max_length=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response


