"""配置管理模块 - 支持环境变量和 YAML/JSON 配置文件"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# YAML 支持（可选依赖）
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class LLMConfig:
    """LLM 相关配置"""
    api_base: str = ""
    api_key: str = "EMPTY"
    model: str = "QuantTrio/MiniMax-M2-AWQ"
    
    # 请求参数
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 60.0


@dataclass
class RunnerConfig:
    """Runner 运行时配置"""
    show_thinking: bool = False
    show_request: bool = False
    # 日志详细级别: minimal | normal | verbose
    # - minimal: 只显示关键信息（模型、工具调用名称）
    # - normal: 显示主要内容（默认）
    # - verbose: 显示完整详细信息
    log_level: str = "normal"
    # 注意：max_iterations 应该在 Agent 级别配置，不在 Runner 级别
    # 这符合 Google ADK 的设计理念：Agent 是配置，Runner 是无状态执行引擎


@dataclass
class Config:
    """
    主配置类 - 管理所有配置项
    
    配置优先级（从高到低）:
    1. 代码中直接传入的参数
    2. 环境变量
    3. 配置文件
    4. 默认值
    
    环境变量命名规则:
    - LLM 配置: TINY_ADK_API_BASE, TINY_ADK_API_KEY, TINY_ADK_MODEL
    - Runner 配置: TINY_ADK_SHOW_THINKING, TINY_ADK_SHOW_REQUEST
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    
    # 环境变量前缀
    ENV_PREFIX: str = "TINY_ADK_"
    
    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        env_prefix: str = "TINY_ADK_",
    ) -> "Config":
        """
        加载配置
        
        Args:
            config_file: 可选的配置文件路径 (支持 .yaml, .yml, .json)
            env_prefix: 环境变量前缀
            
        Returns:
            Config 实例
        """
        config = cls()
        config.ENV_PREFIX = env_prefix
        
        # 1. 从配置文件加载
        if config_file:
            config._load_from_file(config_file)
        else:
            # 尝试自动发现配置文件
            config._auto_discover_config()
        
        # 2. 从环境变量加载（会覆盖配置文件的值）
        config._load_from_env()
        
        return config
    
    def _auto_discover_config(self) -> None:
        """自动发现配置文件"""
        # 查找顺序（优先 YAML）
        search_paths = [
            Path.cwd() / "tiny_adk.yaml",
            Path.cwd() / "tiny_adk.yml",
            Path.cwd() / ".tiny_adk.yaml",
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yml",
            Path.home() / ".tiny_adk.yaml",
            # 也支持 JSON 作为后备
            Path.cwd() / "tiny_adk.json",
            Path.cwd() / ".tiny_adk.json",
        ]
        
        for path in search_paths:
            if path.exists():
                self._load_from_file(path)
                break
    
    def _load_from_file(self, config_file: str | Path) -> None:
        """从配置文件加载（支持 YAML 和 JSON）"""
        path = Path(config_file)
        
        if not path.exists():
            return
        
        suffix = path.suffix.lower()
        
        with open(path, 'r', encoding='utf-8') as f:
            if suffix in ('.yaml', '.yml'):
                if not YAML_AVAILABLE:
                    raise ImportError(
                        "需要安装 PyYAML 来加载 YAML 配置文件: pip install pyyaml"
                    )
                data = yaml.safe_load(f)
            else:
                # 默认使用 JSON
                data = json.load(f)
        
        if data:
            self._apply_dict(data)
    
    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        prefix = self.ENV_PREFIX
        
        # LLM 配置
        if api_base := os.getenv(f"{prefix}API_BASE"):
            self.llm.api_base = api_base
        
        if api_key := os.getenv(f"{prefix}API_KEY"):
            self.llm.api_key = api_key
        
        if model := os.getenv(f"{prefix}MODEL"):
            self.llm.model = model
        
        if temperature := os.getenv(f"{prefix}TEMPERATURE"):
            self.llm.temperature = float(temperature)
        
        if max_tokens := os.getenv(f"{prefix}MAX_TOKENS"):
            self.llm.max_tokens = int(max_tokens)
        
        if timeout := os.getenv(f"{prefix}TIMEOUT"):
            self.llm.timeout = float(timeout)
        
        # Runner 配置
        if show_thinking := os.getenv(f"{prefix}SHOW_THINKING"):
            self.runner.show_thinking = show_thinking.lower() in ('true', '1', 'yes')
        
        if show_request := os.getenv(f"{prefix}SHOW_REQUEST"):
            self.runner.show_request = show_request.lower() in ('true', '1', 'yes')
    
    def _apply_dict(self, data: dict[str, Any]) -> None:
        """从字典应用配置"""
        if llm_data := data.get("llm"):
            if "api_base" in llm_data:
                self.llm.api_base = llm_data["api_base"]
            if "api_key" in llm_data:
                self.llm.api_key = llm_data["api_key"]
            if "model" in llm_data:
                self.llm.model = llm_data["model"]
            if "temperature" in llm_data:
                self.llm.temperature = llm_data["temperature"]
            if "max_tokens" in llm_data:
                self.llm.max_tokens = llm_data["max_tokens"]
            if "timeout" in llm_data:
                self.llm.timeout = llm_data["timeout"]
        
        if runner_data := data.get("runner"):
            if "show_thinking" in runner_data:
                self.runner.show_thinking = runner_data["show_thinking"]
            if "show_request" in runner_data:
                self.runner.show_request = runner_data["show_request"]
            if "log_level" in runner_data:
                self.runner.log_level = runner_data["log_level"]
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "llm": {
                "api_base": self.llm.api_base,
                "api_key": self.llm.api_key,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
            },
            "runner": {
                "show_thinking": self.runner.show_thinking,
                "show_request": self.runner.show_request,
                "log_level": self.runner.log_level,
            },
        }
    
    def save(self, config_file: str | Path) -> None:
        """保存配置到文件（根据扩展名自动选择格式）"""
        path = Path(config_file)
        suffix = path.suffix.lower()
        
        with open(path, 'w', encoding='utf-8') as f:
            if suffix in ('.yaml', '.yml'):
                if not YAML_AVAILABLE:
                    raise ImportError(
                        "需要安装 PyYAML 来保存 YAML 配置文件: pip install pyyaml"
                    )
                yaml.dump(
                    self.to_dict(), 
                    f, 
                    default_flow_style=False, 
                    allow_unicode=True,
                    sort_keys=False,
                )
            else:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# 全局默认配置实例（懒加载）
_default_config: Config | None = None


def get_config() -> Config:
    """获取全局默认配置"""
    global _default_config
    if _default_config is None:
        _default_config = Config.load()
    return _default_config


def set_config(config: Config) -> None:
    """设置全局默认配置"""
    global _default_config
    _default_config = config

