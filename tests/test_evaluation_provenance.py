import json
import sys
import tempfile
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluation_provenance import sample_labels, finite_json
class ProvenanceTests(unittest.TestCase):
    def test_missing_metadata_does_not_invent_trial_names(self):
        self.assertEqual(sample_labels(None,2),[('pair_0','unverified'),('pair_1','unverified')])
    def test_single_pair_rejected(self):
        with self.assertRaises(ValueError): sample_labels(None,1)
    def test_metadata_counts_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.json';p.write_text(json.dumps({'samples':[{'trial':'heldout','frame_idx':1,'gesture':'G1'},{'trial':'heldout','frame_idx':2,'gesture':'G2'}]}))
            self.assertEqual(sample_labels(p,2)[0],('heldout f=1','G1'))
            with self.assertRaises(ValueError): sample_labels(p,3)
    def test_duplicate_samples_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.json';p.write_text(json.dumps({'samples':[dict(trial='t',frame_idx=1,gesture='G1')]*2}))
            with self.assertRaises(ValueError): sample_labels(p,2)

    def test_nonfinite_metrics_are_strict_json_null(self):
        cleaned = finite_json({'matrix': [[float('inf'), float('nan')], [3.4, 0.0]]})
        self.assertEqual(json.loads(json.dumps(cleaned, allow_nan=False)), {'matrix': [[None, None], [3.4, 0.0]]})
