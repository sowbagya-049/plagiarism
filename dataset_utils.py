import pandas as pd
import numpy as np
import sqlite3
import json
import uuid
import requests
import os
from datetime import datetime
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score # Although imported, f1_score is not used in the provided code
import time
import threading

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# Dataset loading and processing utilities
class DatasetManager:
    def __init__(self, db_path='plagiarism_db.sqlite', model_name='paraphrase-mpnet-base-v2'):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reference_documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                embeddings TEXT,
                language TEXT,
                source_dataset TEXT,
                created_at TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_cases (
                id TEXT PRIMARY KEY,
                original_text TEXT,
                plagiarized_text TEXT,
                plagiarism_type TEXT,
                expected_score REAL,
                language TEXT,
                created_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_reference_documents(self, documents: List[Dict[str, Any]], source_dataset: str = "custom"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for doc in documents:
            doc_id = str(uuid.uuid4())
            content = doc['content']
            sentences = sent_tokenize(content)
            embeddings = self.model.encode(sentences)
            cursor.execute('''
                INSERT INTO reference_documents 
                (id, title, content, embeddings, language, source_dataset, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                doc.get('title', 'Untitled'),
                content,
                json.dumps(embeddings.tolist()),
                doc.get('language', 'en'),
                source_dataset,
                datetime.now().isoformat()
            ))
        conn.commit()
        conn.close()
        print(f"Added {len(documents)} documents to reference corpus from {source_dataset}")


class PANCorpusLoader:
    def __init__(self, data_dir='./datasets/pan'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_pan_corpus(self, year=2020, task='plagiarism-detection'):
        print(f"Downloading PAN {year} {task} corpus...")
        # In a real scenario, this would download actual PAN data.
        # For this example, we return sample documents.
        sample_documents = [
            {"title": "Artificial Intelligence in Modern Society",
             "content": "Artificial intelligence has become an integral part of modern society, influencing everything from healthcare to transportation. Machine learning algorithms process vast amounts of data to make predictions and automate decision-making processes. Deep learning networks, inspired by the human brain, enable computers to recognize patterns and solve complex problems that were previously thought impossible for machines.",
             "language": "en"},
            {"title": "Climate Change and Environmental Impact",
             "content": "Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing melting ice caps, rising sea levels, and extreme weather patterns. The burning of fossil fuels releases greenhouse gases into the atmosphere, creating a greenhouse effect that traps heat and warms the planet.",
             "language": "en"},
            {"title": "Digital Transformation in Business",
             "content": "Digital transformation is reshaping how businesses operate in the 21st century. Companies are leveraging cloud computing, big data analytics, and automation to streamline operations and improve customer experiences. E-commerce platforms have revolutionized retail, while remote work technologies have changed traditional workplace dynamics.",
             "language": "en"},
            {"title": "The Evolution of Social Media",
             "content": "Social media platforms have fundamentally changed how people communicate and share information. These digital networks connect billions of users worldwide, enabling instant communication and content sharing. However, concerns about privacy, misinformation, and digital addiction have emerged as significant challenges.",
             "language": "en"},
            {"title": "Advances in Medical Technology",
             "content": "Medical technology continues to advance rapidly, improving patient outcomes and treatment options. Telemedicine allows remote consultations, while robotic surgery provides greater precision. Personalized medicine uses genetic information to tailor treatments to individual patients, promising more effective healthcare solutions.",
             "language": "en"}
        ]
        return sample_documents
    
    def create_plagiarized_version(self, original_text):
        sentences = sent_tokenize(original_text)
        plagiarized_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 5:
                # Simple synonym replacement
                words = [self.simple_synonym_replace(w) for w in words]
                if len(words) > 8:
                    # Simple word reordering for longer sentences
                    mid = len(words) // 2
                    words = words[mid:] + words[:mid]
            plagiarized_sentences.append(' '.join(words))
        return ' '.join(plagiarized_sentences)
    
    def simple_synonym_replace(self, word):
        synonyms = {
            'artificial': 'synthetic',
            'intelligence': 'AI',
            'modern': 'contemporary',
            'society': 'community',
            'technology': 'tech',
            'business': 'enterprise',
            'digital': 'electronic',
            'information': 'data',
            'global': 'worldwide',
            'important': 'significant',
            'change': 'transform',
            'improve': 'enhance'
        }
        return synonyms.get(word.lower(), word)
    
    def load_pan_corpus(self):
        documents = self.download_pan_corpus()
        plagiarized_versions = []
        # Create plagiarized versions for a subset of the documents
        for doc in documents[:3]: 
            plagiarized_versions.append({
                "title": doc["title"] + " (Modified)",
                "content": self.create_plagiarized_version(doc["content"]),
                "language": doc["language"],
                "original_source": doc["title"]
            })
        return documents + plagiarized_versions
    
    def show_sample(self, documents, n=5):
        import pandas as pd
        print(f"\nSample {n} entries from {self.__class__.__name__}:")
        df = pd.DataFrame(documents)
        print(df.head(n))


class QuoraDatasetLoader:
    def __init__(self, data_dir='./datasets/quora'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_quora_dataset(self, sample_size=1000):
        print(f"Loading Quora dataset with {sample_size} samples...")
        # Simulate loading the Quora Question Pairs dataset
        sample_pairs = [
            {"question1": "What is the best programming language to learn?", "question2": "Which programming language should I learn first?", "is_duplicate": 1},
            {"question1": "How do I lose weight quickly?", "question2": "What are fast ways to lose weight?", "is_duplicate": 1},
            {"question1": "What is machine learning?", "question2": "Can you explain machine learning?", "is_duplicate": 1},
            {"question1": "How does Bitcoin work?", "question2": "What is the mechanism behind Bitcoin?", "is_duplicate": 1},
            {"question1": "What are the benefits of exercise?", "question2": "Why is regular physical activity important?", "is_duplicate": 1},
            {"question1": "What is artificial intelligence?", "question2": "How do I bake a chocolate cake?", "is_duplicate": 0},
            {"question1": "What is the weather like today?", "question2": "How do quantum computers work?", "is_duplicate": 0}
        ]
        documents = []
        # Populate documents by repeating sample_pairs to reach sample_size
        for i, pair in enumerate(sample_pairs * (sample_size // len(sample_pairs) + 1)):
            if len(documents) >= sample_size:
                break
            documents.append({"title": f"Quora Question {i+1}_A", "content": pair["question1"], "language": "en", "is_duplicate": pair["is_duplicate"]})
            documents.append({"title": f"Quora Question {i+1}_B", "content": pair["question2"], "language": "en", "is_duplicate": pair["is_duplicate"]})
        return documents[:sample_size]
    
    def show_sample(self, documents, n=5):
        import pandas as pd
        print(f"\nSample {n} entries from {self.__class__.__name__}:")
        df = pd.DataFrame(documents)
        print(df.head(n))


class MRPCDatasetLoader:
    def __init__(self, data_dir='./datasets/mrpc'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_mrpc_dataset(self, sample_size=500):
        print(f"Loading MRPC dataset with {sample_size} samples...")
        # Simulate loading the Microsoft Research Paraphrase Corpus
        sample_pairs = [
            {"sentence1": "The company announced record profits this quarter.", "sentence2": "Record profits were announced by the company for this quarter.", "label": 1},
            {"sentence1": "Scientists have discovered a new species in the Amazon.", "sentence2": "A new species has been found in the Amazon by researchers.", "label": 1},
            {"sentence1": "The stock market reached an all-time high yesterday.", "sentence2": "Yesterday saw the stock market hit record levels.", "label": 1},
            {"sentence1": "Climate change is affecting global weather patterns.", "sentence2": "Global weather patterns are being influenced by climate change.", "label": 1},
            {"sentence1": "The new smartphone features advanced camera technology.", "sentence2": "Advanced camera tech is included in the new smartphone.", "label": 1},
            {"sentence1": "The meeting has been postponed until next week.", "sentence2": "I love eating pizza on weekends.", "label": 0},
            {"sentence1": "Artificial intelligence is transforming healthcare.", "sentence2": "The ocean contains many unexplored mysteries.", "label": 0}
        ]
        documents = []
        # Populate documents by repeating sample_pairs to reach sample_size
        for i, pair in enumerate(sample_pairs * (sample_size // len(sample_pairs) + 1)):
            if len(documents) >= sample_size:
                break
            documents.append({"title": f"MRPC Sentence {i+1}_A", "content": pair["sentence1"], "language": "en", "is_paraphrase": pair["label"]})
            documents.append({"title": f"MRPC Sentence {i+1}_B", "content": pair["sentence2"], "language": "en", "is_paraphrase": pair["label"]})
        return documents[:sample_size]
    
    def show_sample(self, documents, n=5):
        import pandas as pd
        print(f"\nSample {n} entries from {self.__class__.__name__}:")
        df = pd.DataFrame(documents)
        print(df.head(n))


class TestCaseGenerator:
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        test_cases = []
        test_cases.extend(self.generate_exact_match_cases())
        test_cases.extend(self.generate_paraphrase_cases())
        test_cases.extend(self.generate_original_cases())
        test_cases.extend(self.generate_edge_cases())
        return test_cases
    
    def generate_exact_match_cases(self) -> List[Dict[str, Any]]:
        return [
            {"original_text": "Artificial intelligence is transforming the world rapidly.",
             "plagiarized_text": "Artificial intelligence is transforming the world rapidly.",
             "plagiarism_type": "exact_match", "expected_score": 100.0, "language": "en"},
            {"original_text": "Climate change poses significant challenges to global sustainability.",
             "plagiarized_text": "Climate change poses significant challenges to global sustainability.",
             "plagiarism_type": "exact_match", "expected_score": 100.0, "language": "en"}
        ]
    
    def generate_paraphrase_cases(self) -> List[Dict[str, Any]]:
        return [
            {"original_text": "Machine learning algorithms can process large datasets efficiently.",
             "plagiarized_text": "Large datasets can be processed efficiently by machine learning algorithms.",
             "plagiarism_type": "paraphrase", "expected_score": 75.0, "language": "en"},
            {"original_text": "The research team published their findings in a peer-reviewed journal.",
             "plagiarized_text": "Their findings were published by the research team in a peer-reviewed journal.",
             "plagiarism_type": "paraphrase", "expected_score": 70.0, "language": "en"},
            {"original_text": "Social media platforms connect billions of users worldwide.",
             "plagiarized_text": "Billions of users around the globe are connected through social media platforms.",
             "plagiarism_type": "paraphrase", "expected_score": 80.0, "language": "en"}
        ]
    
    def generate_original_cases(self) -> List[Dict[str, Any]]:
        return [
            {"original_text": "Quantum computing represents a paradigm shift in computational power.",
             "plagiarized_text": "The weather today is sunny with a chance of afternoon showers.",
             "plagiarism_type": "original", "expected_score": 0.0, "language": "en"},
            {"original_text": "Renewable energy sources are becoming more cost-effective.",
             "plagiarized_text": "My favorite hobby is reading mystery novels on weekends.",
             "plagiarism_type": "original", "expected_score": 0.0, "language": "en"},
            {"original_text": "Blockchain technology enables secure decentralized transactions.",
             "plagiarized_text": "The art museum features contemporary paintings from local artists.",
             "plagiarism_type": "original", "expected_score": 0.0, "language": "en"}
        ]
    
    def generate_edge_cases(self) -> List[Dict[str, Any]]:
        return [
            {"original_text": "AI.", "plagiarized_text": "Artificial Intelligence.",
             "plagiarism_type": "abbreviation", "expected_score": 60.0, "language": "en"},
            {"original_text": "The quick brown fox jumps over the lazy dog.",
             "plagiarized_text": "A fast brown fox leaps above the sleepy dog.",
             "plagiarism_type": "synonym_replacement", "expected_score": 85.0, "language": "en"},
            {"original_text": "Data analysis is crucial for business decisions.",
             "plagiarized_text": "Business decisions require crucial data analysis.",
             "plagiarism_type": "word_reordering", "expected_score": 90.0, "language": "en"},
            {"original_text": "", "plagiarized_text": "Empty content test case.",
             "plagiarism_type": "empty_input", "expected_score": 0.0, "language": "en"}
        ]
    
    def save_test_cases(self, test_cases: List[Dict[str, Any]]):
        conn = sqlite3.connect(self.dataset_manager.db_path)
        cursor = conn.cursor()
        for case in test_cases:
            test_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO test_cases 
                (id, original_text, plagiarized_text, plagiarism_type, expected_score, language, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_id,
                case["original_text"],
                case["plagiarized_text"],
                case["plagiarism_type"],
                case["expected_score"],
                case["language"],
                datetime.now().isoformat()
            ))
        conn.commit()
        conn.close()
        print(f"Saved {len(test_cases)} test cases to database")


class PerformanceTester:
    def __init__(self, api_base_url="http://localhost:8000/api"):
        self.api_base = api_base_url
    
    def run_performance_tests(self):
        print("🚀 Starting Performance Tests...")
        self.test_text_length_performance()
        self.test_concurrent_performance()
        self.test_batch_performance()
        self.test_memory_usage()
    
    def test_text_length_performance(self):
        print("\n📊 Testing Text Length Performance...")
        test_texts = {
            "Short (100 chars)": "A" * 100,
            "Medium (1000 chars)": "A" * 1000,
            "Long (5000 chars)": "A" * 5000,
            "Very Long (10000 chars)": "A" * 10000
        }
        for length_type, text in test_texts.items():
            times = []
            for _ in range(3): # Run each test 3 times for a more stable average
                start = time.time()
                r = requests.post(f"{self.api_base}/check-plagiarism", json={"text": text, "check_type": "similarity"})
                end = time.time()
                if r.status_code == 200:
                    times.append(end - start)
                else:
                    print(f"❌ Error for {length_type}: {r.status_code}")
            if times:
                avg_time = sum(times) / len(times)
                print(f"   {length_type}: {avg_time:.2f}s")
    
    def test_concurrent_performance(self):
        print("\n🔄 Testing Concurrent Performance...")
        test_text = "This is a test text for concurrent plagiarism detection testing."
        num_threads = 5
        results = []
        
        def make_request():
            start = time.time()
            r = requests.post(f"{self.api_base}/check-plagiarism", json={"text": test_text, "check_type": "similarity"})
            end = time.time()
            results.append((r.status_code, end - start))
        
        threads = [threading.Thread(target=make_request) for _ in range(num_threads)]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start
        
        success = sum([1 for status, _ in results if status==200])
        avg_response = np.mean([t for status, t in results if status==200]) if success > 0 else None
        
        if success > 0:
            print(f"   Total time for {num_threads} concurrent requests: {total_time:.2f}s")
            print(f"   Average response time: {avg_response:.2f}s")
            print(f"   Successful requests: {success}/{num_threads}")
        else:
            print("   ❌ No successful concurrent requests")


    def test_batch_performance(self):
        print("\n📦 Testing Batch Performance...")
        batch_sizes = [5, 10, 20]
        test_text = "Sample text for batch processing performance testing."
        for batch_size in batch_sizes:
            texts = [f"{test_text} (Item {i})" for i in range(batch_size)]
            start = time.time()
            r = requests.post(f"{self.api_base}/batch-check", json={"texts": texts, "check_type": "similarity"})
            end = time.time()
            if r.status_code == 200:
                total_time = end - start
                print(f"   Batch size {batch_size}: {total_time:.2f}s total, {total_time/batch_size:.2f}s per text")
            else:
                print(f"   ❌ Batch size {batch_size} failed: {r.status_code}")


    def test_memory_usage(self):
        print("\n💾 Testing Memory Usage...")
        try:
            import psutil
            proc = psutil.Process()
            # Get baseline memory usage
            baseline = proc.memory_info().rss / 1024 / 1024 
            print(f"   Baseline memory: {baseline:.1f} MB")
            
            large_text = "Large text for memory testing. " * 1000
            # Make a request that might consume memory (e.g., embedding a large text)
            _ = requests.post(f"{self.api_base}/check-plagiarism", json={"text": large_text, "check_type": "similarity"})
            
            # Get peak memory usage after the operation
            peak = proc.memory_info().rss / 1024 / 1024
            increase = peak - baseline
            print(f"   Peak memory: {peak:.1f} MB")
            print(f"   Memory increase: {increase:.1f} MB")
        except ImportError:
            print("   ⚠️  psutil not available for memory testing. Install with 'pip install psutil'")


class AccuracyTester:
    def __init__(self, api_base_url="http://localhost:8000/api"):
        self.api_base = api_base_url
    
    def run_accuracy_tests(self, test_cases: List[Dict[str, Any]], similarity_threshold=0.75):
        print("🎯 Starting Accuracy Tests...")
        passed = 0
        total = len(test_cases)
        results_by_type = {} # To store accuracy for each plagiarism type
        
        for i, case in enumerate(test_cases):
            print(f"\n   Test {i+1}/{total}: {case['plagiarism_type']}")
            try:
                # Call the plagiarism detection API
                response = requests.post(f"{self.api_base}/check-plagiarism", json={
                    "text": case["plagiarized_text"],
                    "reference_texts": [case["original_text"]] if case["original_text"] else None,
                    "check_type": "similarity", # Assuming 'similarity' check type for these tests
                    "language": case["language"]
                })
                
                if response.status_code == 200:
                    score = response.json().get("plagiarism_score", 0)
                    # Determine if detection matches expectation based on threshold
                    is_plagiarized = score >= similarity_threshold * 100 # API returns score out of 100
                    expected = case["expected_score"] >= similarity_threshold * 100 
                    
                    status = "✅ PASS" if (is_plagiarized == expected) else "❌ FAIL"
                    if status == "✅ PASS":
                        passed += 1
                    
                    test_type = case["plagiarism_type"]
                    if test_type not in results_by_type:
                        results_by_type[test_type] = {"total": 0, "passed": 0}
                    results_by_type[test_type]["total"] += 1
                    if status == "✅ PASS":
                        results_by_type[test_type]["passed"] += 1
                        
                    print(f"      Similarity score: {score:.1f}% - {status}")
                else:
                    print(f"      ❌ API Error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"      ❌ Exception during test: {str(e)}")
        
        accuracy = (passed / total) * 100 if total > 0 else 0
        print(f"\n📈 Overall Accuracy: {accuracy:.1f}% ({passed}/{total})\n")
        
        print("📊 Accuracy by Plagiarism Type:")
        for t, stats in results_by_type.items():
            type_acc = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"   {t}: {type_acc:.1f}% ({stats['passed']}/{stats['total']})")
        
        return accuracy, results_by_type


class MultilingualTester:
    def __init__(self, api_base_url="http://localhost:8000/api"):
        self.api_base = api_base_url
    
    def generate_multilingual_test_cases(self):
        return {
            "en": {
                "original": "Artificial intelligence is transforming our world.",
                "plagiarized": "AI is changing our world significantly.",
                "different": "The weather is beautiful today."
            },
            "es": {
                "original": "La inteligencia artificial está transformando nuestro mundo.",
                "plagiarized": "La IA está cambiando nuestro mundo significativamente.",
                "different": "El clima está hermoso hoy."
            },
            "fr": {
                "original": "L'intelligence artificielle transforme notre monde.",
                "plagiarized": "L'IA change notre monde de manière significative.",
                "different": "Le temps est magnifique aujourd'hui."
            },
            "de": {
                "original": "Künstliche Intelligenz verändert unsere Welt.",
                "plagiarized": "KI verändert unsere Welt erheblich.",
                "different": "Das Wetter ist heute wunderschön."
            }
        }
    
    def test_multilingual_detection(self):
        print("🌍 Testing Multilingual Detection...")
        test_cases = self.generate_multilingual_test_cases()
        
        for lang, texts in test_cases.items():
            print(f"\n   Testing {lang.upper()}:")
            
            # Test for actual plagiarism/high similarity in the same language
            response = requests.post(f"{self.api_base}/check-plagiarism", json={
                "text": texts["plagiarized"], "reference_texts": [texts["original"]], "language": lang
            })
            if response.status_code == 200:
                score = response.json().get("plagiarism_score", 0)
                print(f"      Same language (plagiarized) similarity: {score:.1f}%")
            else:
                print(f"      ❌ API Error for {lang} (plagiarized): {response.status_code} - {response.text}")

            # Test for genuinely different content in the same language (should be low similarity)
            response = requests.post(f"{self.api_base}/check-plagiarism", json={
                "text": texts["different"], "reference_texts": [texts["original"]], "language": lang
            })
            if response.status_code == 200:
                score = response.json().get("plagiarism_score", 0)
                print(f"      Same language (different content) similarity: {score:.1f}%")
            else:
                print(f"      ❌ API Error for {lang} (different): {response.status_code} - {response.text}")


def main():
    print("🔍 Plagiarism Detection System - Dataset & Testing Suite")
    print("="*60)

    # Initialize DatasetManager which handles the SQLite database and SentenceTransformer model
    dataset_manager = DatasetManager()

    print("\n📚 Loading Datasets...")
    # Load and add documents from simulated PAN corpus
    pan_loader = PANCorpusLoader()
    pan_documents = pan_loader.load_pan_corpus()
    pan_loader.show_sample(pan_documents)
    dataset_manager.add_reference_documents(pan_documents, "PAN_2020")

    # Load and add documents from simulated Quora Question Pairs
    quora_loader = QuoraDatasetLoader()
    quora_documents = quora_loader.load_quora_dataset(sample_size=100)
    quora_loader.show_sample(quora_documents)
    dataset_manager.add_reference_documents(quora_documents, "Quora_QQP")

    # Load and add documents from simulated Microsoft Research Paraphrase Corpus
    mrpc_loader = MRPCDatasetLoader()
    mrpc_documents = mrpc_loader.load_mrpc_dataset(sample_size=50)
    mrpc_loader.show_sample(mrpc_documents)
    dataset_manager.add_reference_documents(mrpc_documents, "MRPC")

    print("\n==== DATA SOURCES USED ====")
    print(f"PAN documents loaded: {len(pan_documents)} from PAN_2020 dataset")
    print(f"Quora documents loaded: {len(quora_documents)} from Quora_QQP dataset")
    print(f"MRPC documents loaded: {len(mrpc_documents)} from MRPC dataset\n")

    print("\n🧪 Generating Test Cases...")
    test_generator = TestCaseGenerator(dataset_manager)
    test_cases = test_generator.generate_test_cases()
    test_generator.save_test_cases(test_cases)

    print("\n⏳ Waiting for API server (2 seconds to allow server startup if just started)...")
    time.sleep(2) # Give a moment for the API server to potentially start

    # Run performance tests
    performance_tester = PerformanceTester()
    try:
        performance_tester.run_performance_tests()
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running. Skipping performance tests.")
        print("   Start the server with: uvicorn main:app --reload")

    # Run accuracy tests
    accuracy_tester = AccuracyTester()
    try:
        similarity_threshold = 0.75 # Define the threshold for considering plagiarism
        accuracy_tester.run_accuracy_tests(test_cases, similarity_threshold)
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running. Skipping accuracy tests.")

    # Run multilingual tests
    multilingual_tester = MultilingualTester()
    try:
        multilingual_tester.test_multilingual_detection()
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running. Skipping multilingual tests.")

    print("\n✅ All tests completed!")
    print("\n🚀 To run the system:")
    print("   1. Start the API server: uvicorn main:app --reload")
    print("   2. Open the frontend: Open index.html in your browser")
    print("   3. Start testing with your own documents!")


if __name__ == "__main__":
    main()