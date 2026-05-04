from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.*")

import math

_SCSCORE_MODEL = None
_SYBA_MODEL = None

def _get_scscore_model():
    global _SCSCORE_MODEL
    if _SCSCORE_MODEL is None:
        from scscore.standalone_model_numpy import SCScorer
        _SCSCORE_MODEL = SCScorer()
        _SCSCORE_MODEL.restore()  # uses default model bundled with the scscore package
    return _SCSCORE_MODEL

def _get_syba_model():
    global _SYBA_MODEL
    if _SYBA_MODEL is None:
        from syba.syba import SybaClassifier
        _SYBA_MODEL = SybaClassifier()
        _SYBA_MODEL.fitDefaultScore()
    return _SYBA_MODEL

class RDKitScore:
    """Node scoring function."""

    def __init__(self, score_function="heavyAtomCount") -> None:
        self.score_function = score_function
        # Normalization constants to bound outputs to [0, 1]
        self._H_MAX = 100.0
        self._W_MAX = 1000.0
        self._HW_MAX = 100000.0

    def __call__(self, node):

        if self.score_function == "sascore":
            meanPrecursorSAS = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    meanPrecursorSAS += sascorer.calculateScore(m)
                except:
                    meanPrecursorSAS += 10.0

            if (
                len(node.precursors_to_expand) == 0
            ):  # TODO ZeroDivisionError: division by zero
                return 0

            meanPrecursorSAS = meanPrecursorSAS / len(node.precursors_to_expand)
            node_value = 1.0 - meanPrecursorSAS / 10.0
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "scscore":
            model = _get_scscore_model()
            meanPrecursorSCS = 0.0
            for p in node.precursors_to_expand:
                try:
                    smi = str(p.molecule)
                    _, score = model.get_score_from_smi(smi)
                    meanPrecursorSCS += score
                except:
                    meanPrecursorSCS += 5.0

            if len(node.precursors_to_expand) == 0:
                return 0

            meanPrecursorSCS = meanPrecursorSCS / len(node.precursors_to_expand)
            node_value = (5.0 - meanPrecursorSCS) / 4.0
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0
            return node_value

        elif self.score_function == "syba":
            model = _get_syba_model()
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    smi = str(p.molecule)
                    total += model.predict(smi=smi)
                except:
                    total += -100.0

            if len(node.precursors_to_expand) == 0:
                return 0

            mean_score = total / len(node.precursors_to_expand)
            node_value = 1.0 / (1.0 + math.exp(-mean_score / 10.0))  # Sigmoid to bound to (0, 1)
            return node_value

        elif self.score_function == "heavyAtomCount":
            totalHeavy = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    totalHeavy += CalcNumHeavyAtoms(m)
                except:
                    totalHeavy += 100.0

            # Map to [0, 1]: higher heavy atom count -> lower score
            node_value = 1.0 - (totalHeavy / self._H_MAX)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "weight":
            totalWeight = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    totalWeight += ExactMolWt(m)
                except:
                    totalWeight += 1000.0

            # Map to [0, 1]: higher weight -> lower score
            node_value = 1.0 - (totalWeight / self._W_MAX)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "weightXsascore":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += ExactMolWt(m) * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            # Smoothly bound to (0, 1]
            node_value = 1.0 / (1.0 + total)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "heavyatomsXsascore":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += CalcNumHeavyAtoms(m) * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            # MCTS only compares nodes relatively so it's fine but not very clean mathematically
            node_value = 1.0 / (1.0 + total)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "heavyatomsXscscore":
            model = _get_scscore_model()
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    _, score = model.get_score_from_smi(str(p.molecule))
                    total += CalcNumHeavyAtoms(m) * score
                except:
                    total += 10000.0

            node_value = 1.0 / (1.0 + total)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value
        elif self.score_function == "heavyatomsXweight":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += CalcNumHeavyAtoms(m) * ExactMolWt(m)
                except:
                    total += self._HW_MAX

            node_value = 1.0 - (total / self._HW_MAX)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "sascoreXscscore":
            model = _get_scscore_model()
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    smi = str(p.molecule)
                    m = Chem.MolFromSmiles(smi)
                    sa = sascorer.calculateScore(m)
                    _, sc = model.get_score_from_smi(smi)
                    total += sa * sc
                except:
                    total += 50.0
            if len(node.precursors_to_expand) == 0:
                return 0

            mean_product = total / len(node.precursors_to_expand)
            node_value = 1.0 / mean_product # 1 at best, 50 at worse
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "WxWxSAS":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += ExactMolWt(m) ** 2 * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            node_value = 1.0 / (1.0 + total)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value
