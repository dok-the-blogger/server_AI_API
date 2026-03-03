import yaml
from pathlib import Path
from typing import Optional

PROFILES_DIR = Path(__file__).parent / "profiles"

_profiles: dict[str, dict] = {}

def load_profiles():
    """Загрузить все профили из директории profiles/"""
    _profiles.clear()
    if not PROFILES_DIR.exists():
        return
    for file in PROFILES_DIR.glob("*.yaml"):
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "name" in data:
                _profiles[data["name"]] = data

def get_profile(name: str) -> Optional[dict]:
    """Получить профиль по имени"""
    return _profiles.get(name)

def get_system_prompt(name: str) -> Optional[str]:
    """Получить системный промпт профиля"""
    profile = _profiles.get(name)
    if profile:
        return profile.get("system_prompt")
    return None

def get_fallback_prompt(name: str) -> Optional[str]:
    """Получить запасной (fallback) промпт профиля для обхода blacklist"""
    profile = _profiles.get(name)
    if profile:
        return profile.get("fallback_system_prompt")
    return None
