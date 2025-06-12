import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, roc_curve, auc
from dataclasses import dataclass
import time
import logging

from core.models.face_recognition import FaceRecognitionSystem
from core.utils.config import Config

@dataclass
class EvaluationMetrics:
    """Store evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    far: float  # False Acceptance Rate
    frr: float  # False Rejection Rate
    avg_processing_time: float
    threshold: float

class AccuracyEvaluator:
    """Simple accuracy evaluation for face recognition system"""
    
    def __init__(self, test_data_dir: str = "test_dataset/"):
        self.test_data_dir = test_data_dir
        self.face_system = FaceRecognitionSystem()
        
    def prepare_test_dataset(self) -> Dict[str, List[str]]:
        """Load test dataset from directory structure"""
        test_data = {}
        
        if not os.path.exists(self.test_data_dir):
            print(f"Creating test dataset directory: {self.test_data_dir}")
            os.makedirs(self.test_data_dir)
            return {}
            
        for person_dir in os.listdir(self.test_data_dir):
            person_path = os.path.join(self.test_data_dir, person_dir)
            if os.path.isdir(person_path):
                images = []
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append(os.path.join(person_path, img_file))
                
                if len(images) >= 2:  # Need at least 2 images per person
                    test_data[person_dir] = images
                    
        print(f"Loaded test data for {len(test_data)} people")
        return test_data
    
    def evaluate_with_threshold(self, threshold: float, test_data: Dict[str, List[str]]) -> Dict:
        """Test system with specific threshold"""
        
        # Save original threshold
        original_threshold = Config.FACE_RECOGNITION_THRESHOLD
        Config.FACE_RECOGNITION_THRESHOLD = threshold
        
        true_labels = []
        predicted_labels = []
        similarity_scores = []
        processing_times = []
        
        print(f"Testing threshold: {threshold:.3f}")
        
        for person_id, images in test_data.items():
            # Use first image as reference
            reference_img = cv2.imread(images[0])
            if reference_img is None:
                continue
                
            # Test with same person (positive tests)
            for test_img_path in images[1:]:
                test_img = cv2.imread(test_img_path)
                if test_img is None:
                    continue
                
                start_time = time.time()
                result = self.face_system.verify_student_images(reference_img, test_img)
                processing_time = time.time() - start_time
                
                if result.get('confidence_score') is not None:
                    true_labels.append(1)  # Same person
                    predicted_labels.append(1 if result['success'] else 0)
                    similarity_scores.append(result['confidence_score'])
                    processing_times.append(processing_time)
                
            # Test with different people (negative tests)
            for other_person, other_images in test_data.items():
                if other_person == person_id:
                    continue
                    
                other_img = cv2.imread(other_images[0])
                if other_img is None:
                    continue
                
                start_time = time.time()
                result = self.face_system.verify_student_images(reference_img, other_img)
                processing_time = time.time() - start_time
                
                if result.get('confidence_score') is not None:
                    true_labels.append(0)  # Different person
                    predicted_labels.append(1 if result['success'] else 0)
                    similarity_scores.append(result['confidence_score'])
                    processing_times.append(processing_time)
        
        # Restore original threshold
        Config.FACE_RECOGNITION_THRESHOLD = original_threshold
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, predicted_labels, 
                                        similarity_scores, processing_times, threshold)
        
        return {
            'metrics': metrics,
            'total_tests': len(true_labels)
        }
    
    def _calculate_metrics(self, true_labels: List[int], predicted_labels: List[int], 
                          similarity_scores: List[float], processing_times: List[float],
                          threshold: float) -> EvaluationMetrics:
        """Calculate performance metrics"""
        
        if len(true_labels) == 0:
            return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, threshold)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Calculate TP, FP, TN, FN
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # FAR and FRR
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            far=far,
            frr=frr,
            avg_processing_time=avg_processing_time,
            threshold=threshold
        )
    
    def test_multiple_thresholds(self, test_data: Dict[str, List[str]]) -> Dict:
        """Test different threshold values"""
        
        print("🔧 Testing Different Thresholds...")
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        results = {}
        best_accuracy = 0
        best_threshold = 0.6
        
        for threshold in thresholds:
            result = self.evaluate_with_threshold(threshold, test_data)
            results[threshold] = result
            
            metrics = result['metrics']
            print(f"  Threshold {threshold}: Accuracy={metrics.accuracy:.3f}, "
                  f"F1={metrics.f1_score:.3f}, FAR={metrics.far:.3f}, FRR={metrics.frr:.3f}")
            
            if metrics.accuracy > best_accuracy:
                best_accuracy = metrics.accuracy
                best_threshold = threshold
        
        return {
            'all_results': results,
            'best_threshold': best_threshold,
            'best_accuracy': best_accuracy,
            'current_threshold': Config.FACE_RECOGNITION_THRESHOLD
        }
    
    def generate_simple_report(self, results: Dict, output_dir: str = "evaluation_results/"):
        """Generate simple evaluation report"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create performance plot
        thresholds = list(results['all_results'].keys())
        accuracies = [results['all_results'][t]['metrics'].accuracy for t in thresholds]
        f1_scores = [results['all_results'][t]['metrics'].f1_score for t in thresholds]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, 'b-o', label='Accuracy', linewidth=2)
        plt.plot(thresholds, f1_scores, 'r-s', label='F1-Score', linewidth=2)
        plt.axvline(x=results['current_threshold'], color='gray', linestyle='--', label='Current Threshold')
        plt.axvline(x=results['best_threshold'], color='green', linestyle='--', label='Best Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Face Recognition Performance vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results to JSON
        json_results = {}
        for threshold, data in results['all_results'].items():
            metrics = data['metrics']
            json_results[str(threshold)] = {
                'accuracy': float(metrics.accuracy),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1_score': float(metrics.f1_score),
                'far': float(metrics.far),
                'frr': float(metrics.frr),
                'avg_processing_time': float(metrics.avg_processing_time)
            }
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump({
                'threshold_analysis': json_results,
                'recommendations': {
                    'current_threshold': float(results['current_threshold']),
                    'best_threshold': float(results['best_threshold']),
                    'best_accuracy': float(results['best_accuracy']),
                    'improvement': f"{((results['best_accuracy'] - 0.7) * 100):+.1f}%"
                }
            }, f, indent=2)
        
        # Generate text report
        report = f"""# Face Recognition Evaluation Report

## Current System
- Model: DeepFace with FaceNet
- Current Threshold: {results['current_threshold']}
- Face Detection: OpenCV Haar Cascades

## Results Summary
- Best Threshold: {results['best_threshold']}
- Best Accuracy: {results['best_accuracy']:.3f}
- Current vs Best: {((results['best_accuracy'] - results['all_results'][results['current_threshold']]['metrics'].accuracy) * 100):+.1f}%

## Detailed Results
| Threshold | Accuracy | F1-Score | FAR | FRR | Time(s) |
|-----------|----------|----------|-----|-----|---------|"""
        
        for threshold in thresholds:
            metrics = results['all_results'][threshold]['metrics']
            report += f"\n| {threshold:.1f} | {metrics.accuracy:.3f} | {metrics.f1_score:.3f} | {metrics.far:.3f} | {metrics.frr:.3f} | {metrics.avg_processing_time:.3f} |"
        
        report += f"""

## Recommendations
1. **Update Config.py**: Change FACE_RECOGNITION_THRESHOLD to {results['best_threshold']}
2. **Expected Improvement**: {((results['best_accuracy'] - results['all_results'][results['current_threshold']]['metrics'].accuracy) * 100):+.1f}% accuracy gain
3. **Consider upgrading**: Face detection from Haar Cascades to MTCNN

## Next Steps
- Test with larger dataset
- Implement face quality assessment
- Add liveness detection improvements
"""
        
        with open(os.path.join(output_dir, 'report.md'), 'w') as f:
            f.write(report)
        
        print(f"📊 Report saved to: {output_dir}")