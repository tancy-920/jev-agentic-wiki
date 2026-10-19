"""Single complete original text representation in premise-audited request wires."""
import hashlib
from copy import deepcopy


def reference_complete_evidence(metadata, evidence):
    """Replace duplicate text by a hash-bound reference; preserve every other field.

    Used ONLY when the judge sends all complete ordered original fragments in its
    system instruction. This is a wire representation change, not a retrieval or
    citation scope change and not a semantic support certificate.
    """
    originals = {c.chunk_id: c.text for c in evidence}
    if len(originals) != len(evidence) or not isinstance(metadata, list):
        raise ValueError('complete distinct evidence metadata required')
    result = []
    for item in metadata:
        if not isinstance(item, dict) or item.get('chunk_id') not in originals:
            raise ValueError('wire evidence has no complete original')
        text = originals[item['chunk_id']]
        if item.get('text') != text:
            raise ValueError('wire original differs from complete evidence')
        value = deepcopy(item)
        del value['text']
        value['complete_original_text_ref'] = {
            'location': 'Complete original fragments in system instruction',
            'chunk_id': item['chunk_id'],
            'characters': len(text),
            'sha256_utf8': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        }
        result.append(value)
    if len(result) != len(evidence) or {i['chunk_id'] for i in result} != set(originals):
        raise ValueError('wire metadata must retain every original exactly once')
    return result
