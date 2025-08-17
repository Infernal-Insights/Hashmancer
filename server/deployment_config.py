"""
Vast.ai Deployment Configuration and Template Management
Optimized configurations for fast worker deployment
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class PerformanceTier(Enum):
    BUDGET = "budget"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    ENTERPRISE = "enterprise"

@dataclass
class DeploymentTemplate:
    name: str
    description: str
    image: str
    gpu_requirements: Dict
    system_requirements: Dict
    pricing: Dict
    startup_script: str
    env_vars: Dict
    performance_tier: PerformanceTier

class VastAiDeploymentConfig:
    """Manages optimized deployment configurations for vast.ai"""
    
    def __init__(self):
        self.templates_dir = os.path.join(os.path.dirname(__file__), 'vast_templates')
        self.load_templates()
        
    def load_templates(self):
        """Load deployment templates from JSON configuration"""
        template_file = os.path.join(self.templates_dir, 'quick_deploy_templates.json')
        
        with open(template_file, 'r') as f:
            self.config = json.load(f)
            
        self.templates = {}
        for template_name, template_data in self.config['hashmancer_templates'].items():
            self.templates[template_name] = DeploymentTemplate(
                name=template_data['name'],
                description=template_data['description'],
                image=template_data['image'],
                gpu_requirements=template_data['gpu_requirements'],
                system_requirements=template_data['system_requirements'],
                pricing=template_data['pricing'],
                startup_script=template_data['startup_script'],
                env_vars=template_data['env_vars'],
                performance_tier=self._get_performance_tier(template_name)
            )
    
    def _get_performance_tier(self, template_name: str) -> PerformanceTier:
        """Determine performance tier from template name"""
        if 'budget' in template_name:
            return PerformanceTier.BUDGET
        elif 'balanced' in template_name:
            return PerformanceTier.BALANCED
        elif 'performance' in template_name:
            return PerformanceTier.PERFORMANCE
        else:
            return PerformanceTier.ENTERPRISE
    
    def get_template(self, template_name: str) -> Optional[DeploymentTemplate]:
        """Get a specific deployment template"""
        return self.templates.get(template_name)
    
    def get_templates_by_tier(self, tier: PerformanceTier) -> List[DeploymentTemplate]:
        """Get all templates for a specific performance tier"""
        return [t for t in self.templates.values() if t.performance_tier == tier]
    
    def get_recommended_template(self, 
                                max_price_per_hour: float,
                                min_gpu_memory: int = 6000,
                                job_priority: str = "normal") -> DeploymentTemplate:
        """Get recommended template based on requirements"""
        
        # Budget constraints
        if max_price_per_hour <= 0.75:
            return self.templates['budget_cracker']
        elif max_price_per_hour <= 2.0:
            return self.templates['balanced_cracker']
        elif max_price_per_hour <= 5.0:
            return self.templates['performance_cracker']
        else:
            return self.templates['multi_gpu_farm']
    
    def get_startup_script_path(self, script_name: str) -> str:
        """Get full path to startup script"""
        return os.path.join(self.templates_dir, script_name)
    
    def get_startup_script_content(self, script_name: str) -> str:
        """Get startup script content"""
        script_path = self.get_startup_script_path(script_name)
        try:
            with open(script_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            # Return default script if specific one not found
            return self._get_default_startup_script()
    
    def _get_default_startup_script(self) -> str:
        """Get default startup script content"""
        return """#!/bin/bash
set -e

echo "🚀 Starting Hashmancer Worker..."

# Update and build
cd /workspace/hashmancer
git pull origin main
cd darkling/build && make -j$(nproc)

# Start worker
python3 worker/worker_main.py --config worker_config.json
"""
    
    def get_optimized_env_vars(self, 
                             template_name: str, 
                             job_id: Optional[str] = None,
                             server_url: Optional[str] = None) -> Dict[str, str]:
        """Get optimized environment variables for a template"""
        
        template = self.get_template(template_name)
        if not template:
            return {}
            
        env_vars = template.env_vars.copy()
        
        # Add runtime variables
        if job_id:
            env_vars['HASHMANCER_JOB_ID'] = job_id
            
        if server_url:
            env_vars['HASHMANCER_SERVER_URL'] = server_url
            
        # Add performance optimizations based on tier
        if template.performance_tier == PerformanceTier.PERFORMANCE:
            env_vars.update({
                'HASHCAT_WORKLOAD_PROFILE': '4',  # Nightmare mode
                'HASHCAT_KERNEL_ACCEL': '1024',
                'HASHCAT_KERNEL_LOOPS': '1024',
                'CUDA_CACHE_MAXSIZE': '4294967296'  # 4GB
            })
        elif template.performance_tier == PerformanceTier.BALANCED:
            env_vars.update({
                'HASHCAT_WORKLOAD_PROFILE': '3',  # High performance
                'CUDA_CACHE_MAXSIZE': '2147483648'  # 2GB
            })
        else:  # Budget
            env_vars.update({
                'HASHCAT_WORKLOAD_PROFILE': '2',  # Economic mode
                'CUDA_CACHE_MAXSIZE': '1073741824'  # 1GB
            })
            
        return env_vars
    
    def get_wordlist_recommendations(self, template_name: str) -> Dict:
        """Get wordlist recommendations for a template"""
        template = self.get_template(template_name)
        if not template:
            return self.config['wordlist_templates']['small']
            
        wordlist_size = template.env_vars.get('WORDLIST_SIZE', 'small')
        return self.config['wordlist_templates'].get(wordlist_size, 
                                                   self.config['wordlist_templates']['small'])
    
    def get_rule_recommendations(self, template_name: str) -> Dict:
        """Get rule recommendations for a template"""
        template = self.get_template(template_name)
        if not template:
            return self.config['rule_templates']['basic']
            
        if template.performance_tier == PerformanceTier.PERFORMANCE:
            return self.config['rule_templates']['comprehensive']
        elif template.performance_tier == PerformanceTier.BALANCED:
            return self.config['rule_templates']['basic']
        else:
            return self.config['rule_templates']['basic']
    
    def generate_launch_spec(self, 
                           template_name: str,
                           job_id: Optional[str] = None,
                           server_url: Optional[str] = None) -> Dict:
        """Generate complete launch specification for vast.ai"""
        
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
            
        # Get startup script content
        startup_script = self.get_startup_script_content(template.startup_script)
        
        # Get optimized environment variables
        env_vars = self.get_optimized_env_vars(template_name, job_id, server_url)
        
        return {
            'image': template.image,
            'env': env_vars,
            'onstart': startup_script,
            'runtype': 'ssh',
            'disk': template.system_requirements['storage_gb'],
            'args': [],
            'client_id': 'hashmancer',
            'gpu_requirements': template.gpu_requirements,
            'system_requirements': template.system_requirements,
            'pricing': template.pricing
        }
    
    def list_all_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict:
        """Get detailed information about a template"""
        template = self.get_template(template_name)
        if not template:
            return {}
            
        wordlist_info = self.get_wordlist_recommendations(template_name)
        rule_info = self.get_rule_recommendations(template_name)
        
        return {
            'name': template.name,
            'description': template.description,
            'performance_tier': template.performance_tier.value,
            'estimated_cost_per_hour': template.pricing['max_price_per_hour'],
            'gpu_requirements': template.gpu_requirements,
            'system_requirements': template.system_requirements,
            'wordlists': {
                'count': len(wordlist_info['files']),
                'size_mb': wordlist_info['total_size_mb'],
                'files': wordlist_info['files']
            },
            'rules': {
                'name': rule_info['name'],
                'description': rule_info['description'],
                'files': rule_info['files']
            },
            'optimizations': list(template.env_vars.keys())
        }

# Global instance for easy access
deployment_config = VastAiDeploymentConfig()