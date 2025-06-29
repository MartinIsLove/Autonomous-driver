from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseControl(ABC):
    """Classe base astratta per i moduli di controllo"""

    @abstractmethod
    def get_actions(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Restituisce le azioni di controllo (es. sterzata, accelerazione).
        """
        pass
