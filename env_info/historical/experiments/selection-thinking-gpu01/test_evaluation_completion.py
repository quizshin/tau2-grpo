import json
import tempfile
import unittest
from pathlib import Path
from evaluation_completion import classify_completion

class CompletionTests(unittest.TestCase):
    def fixture(self, out, errors=False):
        row={'task_id':'a','trial':0,'seed':42}
        (out/'run.json').write_text(json.dumps({'planned':[row]}))
        (out/'trajectories.jsonl').write_text('' if errors else json.dumps(row)+'\n')
        (out/'errors.jsonl').write_text(json.dumps(dict(row,error_type='ContextWindowExceededError'))+'\n' if errors else '')
        (out/'summary.json').write_text(json.dumps(dict(planned_trajectories=1,completed_trajectories=int(not errors),failed_trajectories=int(errors),missing_trajectories=0,metrics_valid=not errors)))
    def test_complete_and_context_error(self):
        for errors in (False,True):
            with tempfile.TemporaryDirectory() as d:
                out=Path(d);self.fixture(out,errors)
                self.assertEqual(classify_completion(out,int(errors))['status'],'completed_with_errors' if errors else 'complete')
    def test_reject_crash_missing_duplicate_corruption(self):
        for mode in ('crash','missing','duplicate','corrupt','summary_missing','wrong_seed','wrong_exit'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                out=Path(d);self.fixture(out)
                if mode=='missing':(out/'trajectories.jsonl').write_text('')
                if mode=='duplicate':(out/'trajectories.jsonl').write_text((out/'trajectories.jsonl').read_text()*2)
                if mode=='corrupt':(out/'summary.json').write_text('{')
                if mode=='summary_missing':(out/'summary.json').unlink()
                if mode=='wrong_seed':(out/'trajectories.jsonl').write_text('{"task_id":"a","trial":0,"seed":43}\n')
                with self.assertRaises(RuntimeError):classify_completion(out,-9 if mode=='crash' else 1 if mode=='wrong_exit' else 0)
if __name__=='__main__':unittest.main()
