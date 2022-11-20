"""Need to log down G_i"""
from types import SimpleNamespace

from plato.clients import simple


class Client(simple.Client):
    """Log down G, and all gamma_i"""

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        report=super().customize_report(report)  
        report.gbound=self.trainer.gbound
        return report
    