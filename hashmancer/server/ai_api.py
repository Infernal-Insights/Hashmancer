"""AI Strategy Engine API endpoints for Hashmancer."""

import json
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException
from .ai_strategy_engine import get_ai_strategy_engine, AIInsights, StrategyRecommendation
from .redis_utils import get_redis


async def api_get_ai_insights(current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get AI insights for current attack state."""
    try:
        ai_engine = get_ai_strategy_engine()
        
        # Use provided state or get current state from Redis
        if not current_state:
            current_state = await _get_current_attack_state()
        
        insights = await ai_engine.get_ai_insights(current_state)
        
        return {
            "success": True,
            "insights": {
                "success_probability": insights.success_probability,
                "recommended_strategy": insights.recommended_strategy,
                "estimated_completion": insights.estimated_completion,
                "pattern_confidence": insights.pattern_confidence,
                "resource_recommendation": insights.resource_recommendation,
                "generated_at": insights.generated_at
            }
        }
        
    except Exception as e:
        logging.error(f"AI insights API error: {e}")
        return {
            "success": False,
            "error": "AI insights unavailable",
            "fallback_strategy": "dictionary"
        }


async def api_get_strategy_recommendation(hash_type: str, current_progress: float = 0.0) -> Dict[str, Any]:
    """Get AI strategy recommendation for specific hash type and progress."""
    try:
        ai_engine = get_ai_strategy_engine()
        
        current_state = {
            "hash_type": hash_type,
            "progress": current_progress,
            "timestamp": 0  # Will be set by the engine
        }
        
        strategy_recommendation = await ai_engine.strategy_engine.recommend_strategy(current_state)
        
        return {
            "success": True,
            "recommendation": {
                "strategy": strategy_recommendation.strategy,
                "confidence": strategy_recommendation.confidence,
                "reasoning": strategy_recommendation.reasoning,
                "expected_improvement": strategy_recommendation.expected_improvement,
                "resource_requirements": strategy_recommendation.resource_requirements
            }
        }
        
    except Exception as e:
        logging.error(f"Strategy recommendation API error: {e}")
        return {
            "success": False,
            "error": "Strategy recommendation unavailable",
            "fallback": {
                "strategy": "dictionary",
                "confidence": 0.5,
                "reasoning": "Fallback due to AI unavailability"
            }
        }


async def api_get_pattern_insights(limit: int = 10) -> Dict[str, Any]:
    """Get AI pattern insights and predictions."""
    try:
        ai_engine = get_ai_strategy_engine()
        redis_client = get_redis()
        
        # Get recent successful patterns from Redis
        recent_patterns = []
        for key in redis_client.scan_iter("found:pattern:*"):
            pattern_data = redis_client.get(key)
            if pattern_data:
                recent_patterns.append(pattern_data.decode())
        
        # Get pattern predictions for recent patterns
        insights = []
        for i, pattern in enumerate(recent_patterns[:limit]):
            try:
                prediction = await ai_engine.pattern_engine.predict_next_patterns(pattern)
                insights.append({
                    "current_pattern": pattern,
                    "next_patterns": prediction.next_patterns,
                    "confidence": prediction.confidence,
                    "context_relevance": prediction.context_relevance
                })
            except Exception as e:
                logging.warning(f"Pattern prediction failed for {pattern}: {e}")
        
        return {
            "success": True,
            "patterns": insights[:limit],
            "total_analyzed": len(recent_patterns)
        }
        
    except Exception as e:
        logging.error(f"Pattern insights API error: {e}")
        return {
            "success": False,
            "error": "Pattern insights unavailable",
            "patterns": []
        }


async def api_get_ai_status() -> Dict[str, Any]:
    """Get AI engine status and model information."""
    try:
        ai_engine = get_ai_strategy_engine()
        status = ai_engine.get_model_status()
        
        return {
            "success": True,
            "ai_available": True,
            "status": status
        }
        
    except Exception as e:
        logging.error(f"AI status API error: {e}")
        return {
            "success": False,
            "ai_available": False,
            "error": str(e),
            "status": {}
        }


async def api_trigger_adaptation() -> Dict[str, Any]:
    """Manually trigger AI adaptation cycle."""
    try:
        ai_engine = get_ai_strategy_engine()
        
        # Get current state and trigger adaptation
        current_state = await _get_current_attack_state()
        insights = await ai_engine.get_ai_insights(current_state)
        
        if await ai_engine._should_adapt(current_state, insights):
            await ai_engine._perform_adaptation(insights)
            return {
                "success": True,
                "message": "Adaptation triggered",
                "new_strategy": insights.recommended_strategy
            }
        else:
            return {
                "success": True,
                "message": "No adaptation needed",
                "current_strategy": insights.recommended_strategy
            }
        
    except Exception as e:
        logging.error(f"Adaptation trigger API error: {e}")
        return {
            "success": False,
            "error": "Adaptation trigger failed"
        }


async def api_get_adaptation_history(limit: int = 50) -> Dict[str, Any]:
    """Get AI adaptation history."""
    try:
        redis_client = get_redis()
        
        # Get adaptation log from Redis
        adaptation_log = redis_client.lrange("ai:adaptation_log", 0, limit - 1)
        
        history = []
        for entry in adaptation_log:
            try:
                event = json.loads(entry.decode())
                history.append(event)
            except Exception:
                continue
        
        return {
            "success": True,
            "adaptations": history,
            "total_entries": len(history)
        }
        
    except Exception as e:
        logging.error(f"Adaptation history API error: {e}")
        return {
            "success": False,
            "error": "Adaptation history unavailable",
            "adaptations": []
        }


async def _get_current_attack_state() -> Dict[str, Any]:
    """Get current attack state from Redis and system metrics."""
    try:
        redis_client = get_redis()
        
        # Get basic system state
        active_jobs = redis_client.llen("jobs") if redis_client.exists("jobs") else 0
        found_count = redis_client.hlen("found:map") if redis_client.exists("found:map") else 0
        
        # Get recent performance metrics (simplified)
        state = {
            "active_jobs": active_jobs,
            "found_count": found_count,
            "timestamp": 0,  # Will be set by AI engine
            "hash_type": "unknown",  # Would be determined from active jobs
            "progress": 0.0,
            "gpu_utilization": 0.5,  # Placeholder
            "success_rate": min(0.1, found_count / max(1, active_jobs * 100)),
            "current_patterns": "?l?l?l?l"  # Placeholder
        }
        
        return state
        
    except Exception as e:
        logging.error(f"Error getting current attack state: {e}")
        return {
            "timestamp": 0,
            "active_jobs": 0,
            "found_count": 0
        }