import json
from pathlib import Path
from service.pipeline.analyze import build_result

RULES=json.loads((Path(__file__).resolve().parents[1]/'service/rules.json').read_text())
FRAMES=[dict(timestamp_sec=t,source_frame_index=i,image=b'jpeg') for i,t in enumerate([0.0,5.0,10.0])]


def describe(type_id):
    return build_result([dict(type_id=type_id,timecode_sec=5.0)],FRAMES,RULES)['violations'][0]['description']


def test_verified_rule_is_quoted_with_its_official_source():
    text=describe('counting_without_client')
    assert '№231' in text and 'п.20' in text and 'https://adilet.zan.kz/' in text
    assert '«Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента.»' in text
    assert 'проверки человеком' in text


def test_every_verified_quote_carries_source_point_and_hash():
    for spec in RULES['types'].values():
        for rule in spec['rules']:
            assert rule['verified'] is True and rule['quote'] and rule['point'] and rule['source_url'].startswith('https://adilet.zan.kz/')
            assert len(rule['text_sha256'])==64


def test_type_without_verified_rule_makes_no_legal_claim():
    assert '«' not in describe('food_drink')
