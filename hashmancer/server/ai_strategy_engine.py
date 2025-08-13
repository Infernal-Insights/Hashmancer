"""AI-Powered Attack Strategy Engine for Hashmancer

This module implements intelligent attack strategy selection using multiple AI models
for pattern recognition, strategy optimization, and resource allocation.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        AutoModel,
        pipeline
    )
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoModel = None
    pipeline = None
    TORCH_AVAILABLE = False
    logging.warning("PyTorch/Transformers unavailable - AI features disabled")

from .redis_utils import get_redis
from .pattern_stats import TOKEN_RE
from hashmancer.utils.event_logger import log_error


class ModelType(Enum):
    """Supported AI model types."""
    PATTERN_PREDICTION = "pattern_prediction"
    STRATEGY_SELECTION = "strategy_selection"  
    RESOURCE_ALLOCATION = "resource_allocation"
    WORDLIST_GENERATION = "wordlist_generation"


@dataclass
class AIInsights:
    """Container for AI-generated insights."""
    success_probability: float
    recommended_strategy: str
    estimated_completion: int  # seconds
    pattern_confidence: float
    resource_recommendation: Dict[str, Any]
    generated_at: float = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = time.time()


@dataclass
class StrategyRecommendation:
    """Strategy recommendation from AI engine."""
    strategy: str
    confidence: float
    reasoning: str
    expected_improvement: float
    resource_requirements: Dict[str, Any]


@dataclass
class PatternPrediction:
    """Password pattern prediction."""
    next_patterns: List[Tuple[str, float]]  # (pattern, probability)
    confidence: float
    context_relevance: float


class AIModelManager:
    """Manages multiple AI models for different tasks."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._select_device(device)
        self.models: Dict[ModelType, Any] = {}
        self.tokenizers: Dict[ModelType, Any] = {}
        self.model_configs: Dict[ModelType, Dict[str, Any]] = {}
        self.redis_client = get_redis()
        
        # Model paths from environment or defaults
        self.model_paths = {
            ModelType.PATTERN_PREDICTION: os.getenv("PATTERN_MODEL_PATH", "distilbert-base-uncased"),
            ModelType.STRATEGY_SELECTION: os.getenv("STRATEGY_MODEL_PATH", "distilgpt2"),
            ModelType.RESOURCE_ALLOCATION: os.getenv("RESOURCE_MODEL_PATH", "distilbert-base-uncased"),
            ModelType.WORDLIST_GENERATION: os.getenv("WORDLIST_MODEL_PATH", "distilgpt2")
        }
        
        self._initialize_models()
    
    def _select_device(self, device: str) -> str:
        """Select optimal device for model inference."""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_models(self):
        """Initialize all AI models."""
        if not TORCH_AVAILABLE:
            logging.warning("PyTorch not available - AI models disabled")
            return
            
        for model_type in ModelType:
            try:
                self._load_model(model_type)
            except Exception as e:
                logging.warning(f"Failed to load {model_type.value} model: {e}")
    
    def _load_model(self, model_type: ModelType):
        """Load a specific model type."""
        model_path = self.model_paths[model_type]
        
        try:
            if model_type in [ModelType.PATTERN_PREDICTION, ModelType.RESOURCE_ALLOCATION]:
                # Use BERT-style models for classification tasks
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path)
            else:
                # Use GPT-style models for generation tasks
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                
                # Set pad token for generation models
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            self.models[model_type] = model
            self.tokenizers[model_type] = tokenizer
            self.model_configs[model_type] = {
                "loaded_at": time.time(),
                "device": self.device,
                "model_path": model_path
            }
            
            logging.info(f"Loaded {model_type.value} model on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to load {model_type.value} model: {e}")
            raise
    
    def is_model_available(self, model_type: ModelType) -> bool:
        """Check if a specific model is loaded and available."""
        return model_type in self.models and self.models[model_type] is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "loaded_models": [model_type.value for model_type in self.models.keys()],
            "model_configs": {k.value: v for k, v in self.model_configs.items()}
        }


class PatternRecognitionEngine:
    """AI-powered password pattern recognition and prediction."""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.redis_client = get_redis()
        self.cache_ttl = 3600  # 1 hour cache
    
    async def predict_next_patterns(self, current_sequence: str, context: Dict[str, Any] = None) -> PatternPrediction:
        """Predict next password patterns based on current sequence."""
        if not self.model_manager.is_model_available(ModelType.PATTERN_PREDICTION):
            return self._fallback_pattern_prediction(current_sequence)
        
        # Check cache first
        cache_key = f"ai:pattern_pred:{hash(current_sequence)}"
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            try:
                cached_data = json.loads(cached_result)
                return PatternPrediction(**cached_data)
            except Exception:
                pass
        
        try:
            # Use AI model for prediction
            model = self.model_manager.models[ModelType.PATTERN_PREDICTION]
            tokenizer = self.model_manager.tokenizers[ModelType.PATTERN_PREDICTION]
            
            # Prepare input
            input_text = f"Pattern sequence: {current_sequence}"
            if context:
                input_text += f" Context: {json.dumps(context)}"
            
            # Tokenize and predict
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Get embeddings for pattern analysis
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Convert embeddings to pattern predictions (simplified for demo)
            predictions = self._embeddings_to_patterns(embeddings, current_sequence)
            
            result = PatternPrediction(
                next_patterns=predictions,
                confidence=0.8,  # Calculate actual confidence
                context_relevance=0.7 if context else 0.5
            )
            
            # Cache result
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(asdict(result)))
            
            return result
            
        except Exception as e:
            logging.error(f"Pattern prediction failed: {e}")
            return self._fallback_pattern_prediction(current_sequence)
    
    def _embeddings_to_patterns(self, embeddings: torch.Tensor, current_sequence: str) -> List[Tuple[str, float]]:
        """Convert model embeddings to pattern predictions (placeholder implementation)."""
        # This is a simplified implementation - in production, this would use
        # a trained model specifically for pattern prediction
        common_patterns = ["?l?l?l?l", "?d?d?d?d", "?u?l?l?l", "?l?l?d?d"]
        
        # Simulate pattern probabilities based on embeddings
        probabilities = torch.softmax(embeddings[0][:len(common_patterns)], dim=0).cpu().numpy()
        
        return [(pattern, float(prob)) for pattern, prob in zip(common_patterns, probabilities)]
    
    def _fallback_pattern_prediction(self, current_sequence: str) -> PatternPrediction:
        """Fallback pattern prediction using existing pattern statistics."""
        # Use existing pattern stats as fallback
        from .pattern_stats import generate_mask
        
        try:
            # Generate patterns using existing statistics
            patterns = []
            for _ in range(4):
                mask = generate_mask(8)  # Generate 8-character masks
                patterns.append((mask, 0.25))  # Equal probability
            
            return PatternPrediction(
                next_patterns=patterns,
                confidence=0.5,  # Lower confidence for fallback
                context_relevance=0.3
            )
        except Exception:
            # Ultimate fallback
            return PatternPrediction(
                next_patterns=[("?l?l?l?l?d?d?d?d", 1.0)],
                confidence=0.3,
                context_relevance=0.1
            )


class StrategySelectionEngine:
    """AI-powered attack strategy selection and optimization."""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.redis_client = get_redis()
        
        # Available strategies
        self.strategies = [
            "dictionary",
            "mask_attack", 
            "combinator",
            "hybrid_wordlist_mask",
            "hybrid_mask_wordlist",
            "rule_based",
            "brute_force"
        ]
    
    async def recommend_strategy(self, current_state: Dict[str, Any]) -> StrategyRecommendation:
        """Recommend optimal attack strategy based on current state."""
        if not self.model_manager.is_model_available(ModelType.STRATEGY_SELECTION):
            return self._fallback_strategy_recommendation(current_state)
        
        try:
            # Analyze current performance
            performance_metrics = await self._analyze_current_performance(current_state)
            
            # Use AI model for strategy selection
            model = self.model_manager.models[ModelType.STRATEGY_SELECTION]
            tokenizer = self.model_manager.tokenizers[ModelType.STRATEGY_SELECTION]
            
            # Prepare prompt for strategy selection
            prompt = self._create_strategy_prompt(current_state, performance_metrics)
            
            # Generate strategy recommendation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            strategy_recommendation = self._parse_strategy_response(response)
            
            return strategy_recommendation
            
        except Exception as e:
            logging.error(f"Strategy selection failed: {e}")
            return self._fallback_strategy_recommendation(current_state)
    
    def _create_strategy_prompt(self, state: Dict[str, Any], performance: Dict[str, Any]) -> str:
        """Create prompt for strategy selection model."""
        return f"""
        Current attack state:
        - Hash type: {state.get('hash_type', 'unknown')}
        - Progress: {state.get('progress', 0)}% complete
        - Success rate: {performance.get('success_rate', 0)}%
        - Time elapsed: {state.get('time_elapsed', 0)} seconds
        - GPU utilization: {state.get('gpu_utilization', 0)}%
        
        Recommend the best attack strategy from: {', '.join(self.strategies)}
        
        Strategy recommendation:"""
    
    def _parse_strategy_response(self, response: str) -> StrategyRecommendation:
        """Parse AI model response into strategy recommendation."""
        # Simple parsing - in production this would be more sophisticated
        response_lower = response.lower()
        
        for strategy in self.strategies:
            if strategy.replace('_', ' ') in response_lower or strategy in response_lower:
                return StrategyRecommendation(
                    strategy=strategy,
                    confidence=0.8,
                    reasoning=f"AI recommended {strategy} based on current state",
                    expected_improvement=0.15,  # 15% improvement estimate
                    resource_requirements={"gpu_cores": 1, "memory_gb": 2}
                )
        
        # Default fallback
        return StrategyRecommendation(
            strategy="dictionary",
            confidence=0.5,
            reasoning="Default strategy selected",
            expected_improvement=0.1,
            resource_requirements={"gpu_cores": 1, "memory_gb": 1}
        )
    
    async def _analyze_current_performance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current attack performance."""
        try:
            # Get performance metrics from Redis
            current_job_id = state.get('job_id')
            if current_job_id:
                job_stats = self.redis_client.hgetall(f"job_stats:{current_job_id}")
                return {
                    "success_rate": float(job_stats.get("success_rate", 0)),
                    "average_time": float(job_stats.get("avg_time", 0)),
                    "gpu_efficiency": float(job_stats.get("gpu_efficiency", 0))
                }
        except Exception:
            pass
        
        return {"success_rate": 0, "average_time": 0, "gpu_efficiency": 0}
    
    def _fallback_strategy_recommendation(self, state: Dict[str, Any]) -> StrategyRecommendation:
        """Fallback strategy recommendation using heuristics."""
        # Simple heuristic-based strategy selection
        hash_type = state.get('hash_type', '').lower()
        progress = state.get('progress', 0)
        
        if progress < 10:
            strategy = "dictionary"
        elif "fast" in hash_type:
            strategy = "mask_attack"
        else:
            strategy = "hybrid_wordlist_mask"
        
        return StrategyRecommendation(
            strategy=strategy,
            confidence=0.6,
            reasoning=f"Heuristic selection based on hash type and progress",
            expected_improvement=0.1,
            resource_requirements={"gpu_cores": 1, "memory_gb": 1}
        )


class AIStrategyEngine:
    """Main AI strategy engine coordinating all AI components."""
    
    def __init__(self, device: str = "auto"):
        self.model_manager = AIModelManager(device)
        self.pattern_engine = PatternRecognitionEngine(self.model_manager)
        self.strategy_engine = StrategySelectionEngine(self.model_manager)
        self.redis_client = get_redis()
        
        self.is_running = False
        self.adaptation_interval = 30  # seconds
    
    async def get_ai_insights(self, current_state: Dict[str, Any]) -> AIInsights:
        """Get comprehensive AI insights for current attack state."""
        try:
            # Get pattern predictions
            current_patterns = current_state.get('current_patterns', '')
            pattern_prediction = await self.pattern_engine.predict_next_patterns(
                current_patterns, current_state
            )
            
            # Get strategy recommendation
            strategy_recommendation = await self.strategy_engine.recommend_strategy(current_state)
            
            # Combine insights
            insights = AIInsights(
                success_probability=pattern_prediction.confidence * strategy_recommendation.confidence,
                recommended_strategy=strategy_recommendation.strategy,
                estimated_completion=self._estimate_completion_time(current_state, strategy_recommendation),
                pattern_confidence=pattern_prediction.confidence,
                resource_recommendation=strategy_recommendation.resource_requirements
            )
            
            # Store insights in Redis for monitoring
            self.redis_client.setex(
                "ai:latest_insights",
                300,  # 5 minutes
                json.dumps(asdict(insights))
            )
            
            return insights
            
        except Exception as e:
            logging.error(f"AI insights generation failed: {e}")
            return self._fallback_insights()
    
    def _estimate_completion_time(self, state: Dict[str, Any], strategy: StrategyRecommendation) -> int:
        """Estimate completion time based on current state and recommended strategy."""
        # Simple estimation - in production this would use historical data and ML
        base_time = state.get('estimated_remaining', 3600)  # 1 hour default
        improvement_factor = 1 - strategy.expected_improvement
        return int(base_time * improvement_factor)
    
    def _fallback_insights(self) -> AIInsights:
        """Fallback insights when AI models are unavailable."""
        return AIInsights(
            success_probability=0.5,
            recommended_strategy="dictionary",
            estimated_completion=3600,
            pattern_confidence=0.3,
            resource_recommendation={"gpu_cores": 1, "memory_gb": 1}
        )
    
    async def start_adaptive_monitoring(self):
        """Start continuous adaptive monitoring and optimization."""
        if self.is_running:
            logging.warning("Adaptive monitoring already running")
            return
        
        self.is_running = True
        logging.info("Starting AI adaptive monitoring")
        
        try:
            while self.is_running:
                await self._adaptation_cycle()
                await asyncio.sleep(self.adaptation_interval)
        except Exception as e:
            logging.error(f"Adaptive monitoring error: {e}")
        finally:
            self.is_running = False
    
    async def stop_adaptive_monitoring(self):
        """Stop adaptive monitoring."""
        self.is_running = False
        logging.info("Stopped AI adaptive monitoring")
    
    async def _adaptation_cycle(self):
        """Single adaptation cycle - analyze and adapt if needed."""
        try:
            # Get current system state
            current_state = await self._get_current_system_state()
            
            # Get AI insights
            insights = await self.get_ai_insights(current_state)
            
            # Check if adaptation is needed
            if await self._should_adapt(current_state, insights):
                await self._perform_adaptation(insights)
                
        except Exception as e:
            logging.error(f"Adaptation cycle error: {e}")
    
    async def _get_current_system_state(self) -> Dict[str, Any]:
        """Get current system state for analysis."""
        try:
            # Gather state from Redis and system metrics
            active_jobs = self.redis_client.llen("jobs")
            
            return {
                "active_jobs": active_jobs,
                "timestamp": time.time(),
                "gpu_utilization": 0.7,  # Placeholder - would get from actual monitoring
                "success_rate": 0.15,   # Placeholder
                "current_patterns": "?l?l?l?l"  # Placeholder
            }
        except Exception:
            return {"timestamp": time.time()}
    
    async def _should_adapt(self, state: Dict[str, Any], insights: AIInsights) -> bool:
        """Determine if adaptation is needed based on performance."""
        # Simple adaptation logic - in production this would be more sophisticated
        current_success_rate = state.get('success_rate', 0)
        return current_success_rate < 0.1 or insights.success_probability > 0.8
    
    async def _perform_adaptation(self, insights: AIInsights):
        """Perform system adaptation based on AI insights."""
        try:
            # Log adaptation event
            adaptation_event = {
                "timestamp": time.time(),
                "recommended_strategy": insights.recommended_strategy,
                "success_probability": insights.success_probability,
                "action": "strategy_adaptation"
            }
            
            self.redis_client.lpush("ai:adaptation_log", json.dumps(adaptation_event))
            self.redis_client.ltrim("ai:adaptation_log", 0, 99)  # Keep last 100 events
            
            logging.info(f"AI adaptation: switching to {insights.recommended_strategy}")
            
        except Exception as e:
            logging.error(f"Adaptation execution failed: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all AI models."""
        return {
            "engine_status": "running" if self.is_running else "stopped",
            "model_info": self.model_manager.get_model_info(),
            "last_adaptation": self.redis_client.get("ai:last_adaptation_time"),
            "adaptation_interval": self.adaptation_interval
        }


# Global AI engine instance
_ai_engine: Optional[AIStrategyEngine] = None


def get_ai_strategy_engine() -> AIStrategyEngine:
    """Get or create the global AI strategy engine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIStrategyEngine()
    return _ai_engine


async def initialize_ai_engine():
    """Initialize the AI strategy engine."""
    engine = get_ai_strategy_engine()
    await engine.start_adaptive_monitoring()
    return engine


async def shutdown_ai_engine():
    """Shutdown the AI strategy engine."""
    global _ai_engine
    if _ai_engine:
        await _ai_engine.stop_adaptive_monitoring()
        _ai_engine = None