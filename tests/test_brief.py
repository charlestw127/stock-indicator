import pytest

import brief


@pytest.fixture
def facts():
    return brief.build_facts(
        results={'symbols': {
            'AAPL': {'1m': {'score': 42.0, 'rank': 2,
                            'factors': {'trend': 0.5, 'momentum': -0.4,
                                        'mean_reversion': 0.1,
                                        'volume_flow': 0.2, 'quality': 0.3},
                            'regime': {'trend': 'trending'}}},
            'MSFT': {'1m': {'score': 30.0, 'rank': 3,
                            'factors': {'trend': 0.3, 'momentum': -0.2,
                                        'mean_reversion': 0.0,
                                        'volume_flow': 0.1, 'quality': 0.1},
                            'regime': {'trend': 'mixed'}}},
        }},
        rec={'holdings': [{'symbol': 'AAPL', 'weight': 12.5, 'score': 42.0},
                          {'symbol': 'MSFT', 'weight': 10.0, 'score': 30.0}],
             'selection_horizon': '1m',
             'cash_weight': 40.0,
             'changes': {'added': ['AAPL'], 'dropped': ['NVDA']},
             'risk': {'beta': 1.05, 'ann_vol': 18.3}},
        market={'risk': 'neutral', 'note': 'SPY below 200dma - trend caution',
                'spy_dist_200dma': -2.4, 'vix_percentile': 71.0,
                'exposure': 0.6})


def test_build_facts_gives_every_item_an_id(facts):
    assert facts
    for fid, row in facts.items():
        assert row['id'] == fid
        assert row['label']


def test_build_facts_excludes_prices(facts):
    blob = str(facts).lower()
    assert 'price' not in blob
    assert 'close' not in blob


def test_build_facts_averages_sleeves(facts):
    # trend 0.5 and 0.3 -> 0.4
    assert facts['sleeve.trend']['value'] == pytest.approx(0.4)
    assert facts['sleeve.momentum']['value'] == pytest.approx(-0.3)


def test_verify_keeps_a_grounded_sentence(facts):
    kept, rejected = brief.verify([{
        'text': 'The recommendation holds 2 names with 40.0% in cash.',
        'cited_ids': ['portfolio.n_names', 'portfolio.cash_pct'],
        'claim_type': 'data'}], facts)
    assert len(kept) == 1
    assert not rejected


def test_verify_drops_external_claims(facts):
    kept, rejected = brief.verify([{
        'text': 'The Fed is expected to cut rates next month.',
        'cited_ids': ['market.risk'], 'claim_type': 'external'}], facts)
    assert not kept
    assert 'outside knowledge' in rejected[0]['reason']


def test_verify_drops_uncited_sentences(facts):
    kept, rejected = brief.verify([{
        'text': 'Things look good.', 'cited_ids': [], 'claim_type': 'data'}],
        facts)
    assert not kept
    assert 'cites no facts' in rejected[0]['reason']


def test_verify_drops_unknown_fact_ids(facts):
    kept, rejected = brief.verify([{
        'text': 'Momentum is fine.', 'cited_ids': ['sleeve.telepathy'],
        'claim_type': 'data'}], facts)
    assert not kept
    assert 'unknown' in rejected[0]['reason']


def test_verify_drops_invented_numbers(facts):
    kept, rejected = brief.verify([{
        'text': 'The portfolio holds 17 names.',
        'cited_ids': ['portfolio.n_names'], 'claim_type': 'data'}], facts)
    assert not kept
    assert 'not in the cited facts' in rejected[0]['reason']


def test_verify_allows_numbers_from_cited_facts_only(facts):
    # 40.0 is the cash weight, but this sentence does not cite it
    kept, rejected = brief.verify([{
        'text': 'Cash is 40.0% of the book.',
        'cited_ids': ['portfolio.n_names'], 'claim_type': 'data'}], facts)
    assert not kept


def test_verify_catches_direction_mismatch(facts):
    """momentum averages -0.3, so calling it supportive must fail."""
    kept, rejected = brief.verify([{
        'text': 'The momentum sleeve is supportive across the watchlist.',
        'cited_ids': ['sleeve.momentum'], 'claim_type': 'reasoning'}], facts)
    assert not kept
    assert 'direction mismatch' in rejected[0]['reason']


def test_verify_accepts_correct_direction(facts):
    kept, _ = brief.verify([{
        'text': 'The momentum sleeve is a drag across the watchlist.',
        'cited_ids': ['sleeve.momentum'], 'claim_type': 'reasoning'}], facts)
    assert len(kept) == 1

    kept, _ = brief.verify([{
        'text': 'The trend sleeve is supportive across the watchlist.',
        'cited_ids': ['sleeve.trend'], 'claim_type': 'reasoning'}], facts)
    assert len(kept) == 1


def test_verify_respects_the_sentence_cap(facts):
    many = [{'text': 'Two names are held.',
             'cited_ids': ['portfolio.n_names'], 'claim_type': 'data'}] * 40
    kept, _ = brief.verify(many, facts, max_sentences=3)
    assert len(kept) == 3


def test_number_tolerance_allows_rounding(facts):
    kept, _ = brief.verify([{
        'text': 'Beta is about 1.06.', 'cited_ids': ['portfolio.beta'],
        'claim_type': 'data'}], facts)
    assert len(kept) == 1


def test_fallback_brief_passes_its_own_verifier(facts):
    sentences = brief.fallback_brief(facts)
    kept, rejected = brief.verify(sentences, facts)
    assert len(kept) == len(sentences)
    assert not rejected


def test_generate_without_llm_is_deterministic_and_clean(facts):
    out = brief.generate(
        results={'symbols': {}}, rec={'holdings': [], 'changes': {}},
        market={'risk': 'on', 'note': 'calm'}, use_llm=False)
    assert out['source'].startswith('deterministic')
    assert out['text']
    assert out['confabulation_rate'] == 0.0
    assert out['facts_hash']


def test_generate_reports_confabulation_rate(monkeypatch, facts):
    """A model that invents numbers must show up in the metric."""
    import llm

    monkeypatch.setattr(llm, 'available', lambda: True)
    monkeypatch.setattr(llm, 'structured', lambda *a, **k: {'sentences': [
        {'text': 'The portfolio holds 2 names.',
         'cited_ids': ['portfolio.n_names'], 'claim_type': 'data'},
        {'text': 'Returns will be 99% next week.',
         'cited_ids': ['portfolio.n_names'], 'claim_type': 'data'},
    ]})
    out = brief.generate(rec={'holdings': [{'symbol': 'A', 'weight': 1,
                                            'score': 1},
                                           {'symbol': 'B', 'weight': 1,
                                            'score': 1}],
                              'changes': {}}, use_llm=True)
    assert out['source'] == 'llm'
    assert out['confabulation_rate'] == pytest.approx(0.5)
    assert len(out['rejected']) == 1


def test_generate_falls_back_when_everything_is_rejected(monkeypatch):
    import llm

    monkeypatch.setattr(llm, 'available', lambda: True)
    monkeypatch.setattr(llm, 'structured', lambda *a, **k: {'sentences': [
        {'text': 'Buy everything now.', 'cited_ids': [],
         'claim_type': 'external'}]})
    out = brief.generate(rec={'holdings': [], 'changes': {}},
                         market={'risk': 'on', 'note': 'x'}, use_llm=True)
    assert 'fallback' in out['source']
    assert out['sentences']


def test_generate_survives_an_unavailable_model(monkeypatch):
    import llm

    monkeypatch.setattr(llm, 'available', lambda: True)

    def boom(*a, **k):
        raise llm.LLMUnavailable('no credentials')

    monkeypatch.setattr(llm, 'structured', boom)
    out = brief.generate(rec={'holdings': [], 'changes': {}},
                         market={'risk': 'on', 'note': 'x'}, use_llm=True)
    assert out['source'].startswith('deterministic')
    assert out['text']
