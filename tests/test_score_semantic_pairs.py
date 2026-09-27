import pytest
from tau3_grpo.analysis.score_semantic_pairs import score


def fixture():
    pairs=[{'id':'ab','a':'a','b':'b','expected':'merge','category':'semantic_positive'},
           {'id':'bc','a':'b','b':'c','expected':'unresolved','category':'uncertain'},
           {'id':'ac','a':'a','b':'c','expected':'separate','category':'negative'}]
    rows=[{**p,'actual':v,'stage':'model','filter_reason':None,
           'forward':{'verdict':v,'error':None},'reverse':{'verdict':v,'error':None}}
          for p,v in zip(pairs,['merge','merge','separate'])]
    return {'pairs':pairs},rows,{k:{'comparison_group':'g'} for k in 'abc'}


def test_unresolved_does_not_inflate_accuracy_or_connect_cliques():
    labels,rows,cases=fixture();s=score(labels,rows,cases)
    assert s['overall']['labeled']==2 and s['overall']['correct']==2 and s['overall']['unscored']==1
    assert len(s['nontransitive_triangles'])==1
    assert [g['members'] for g in s['candidate_maximal_cliques']]==[['a','b'],['b','c']]
    assert s['candidate_maximal_cliques'][1]['contains_unresolved_gold_edge']


def test_missing_pair_abstention_and_false_merge_accounting():
    labels,rows,cases=fixture()
    with pytest.raises(ValueError):score(labels,rows[:-1],cases)
    rows[0]['actual']='abstain';rows[2]['actual']='merge'
    s=score(labels,rows,cases)
    assert s['overall']['correct']==0 and s['overall']['false_merges']==1 and s['overall']['abstained']==1
