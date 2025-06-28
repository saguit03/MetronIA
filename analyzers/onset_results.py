import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, NamedTuple


class OnsetType(Enum):
    CORRECT = "correct"
    LATE = "late"
    EARLY = "early"
    MISSING = "missing"
    EXTRA = "extra"


class OnsetMatch(NamedTuple):
    ref_onset: float
    live_onset: float
    ref_note: str
    live_note: str
    time_adjustment: float
    note_similarity: float
    classification: OnsetType


@dataclass
class OnsetDTWAnalysisResult:
    matches: List[OnsetMatch]
    missing_onsets: List[tuple]
    extra_onsets: List[tuple]
    correct_notes: int

    @property
    def correct_matches(self) -> List[OnsetMatch]:
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_note, m.live_note, m.time_adjustment, m.classification,
                           m.note_similarity)
                for m in self.matches if m.classification == OnsetType.CORRECT]

    @property
    def late_matches(self) -> List[OnsetMatch]:
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_note, m.live_note, m.time_adjustment, m.classification,
                           m.note_similarity)
                for m in self.matches if m.classification == OnsetType.LATE]

    @property
    def early_matches(self) -> List[OnsetMatch]:
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_note, m.live_note, m.time_adjustment, m.classification,
                           m.note_similarity)
                for m in self.matches if m.classification == OnsetType.EARLY]

    def to_json_dict(self, mutation_category: str = "", mutation_name: str = "",
                     reference_path: str = "", live_path: str = "",
                     additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        matches_data = []
        for match in self.matches:
            matches_data.append({
                'ref_onset': float(match.ref_onset),
                'live_onset': float(match.live_onset),
                'ref_note': str(match.ref_note),
                'live_note': str(match.live_note),
                'time_adjustment': float(match.time_adjustment),
                'classification': match.classification.value,
                'note_similarity': float(match.note_similarity)
            })

        return {
            'metadata': {
                'analysis_type': 'OnsetDTWAnalysis',
                'mutation_category': mutation_category,
                'mutation_name': mutation_name,
                'reference_path': reference_path,
                'live_path': live_path,
            },
            'matches': matches_data,
            'missing_onsets': [{'onset': float(t), 'pitch': float(p)} for t, p in self.missing_onsets],
            'extra_onsets': [{'onset': float(t), 'pitch': float(p)} for t, p in self.extra_onsets],
        }

    def export_to_json(self, filepath: str, mutation_category: str = "", mutation_name: str = "",
                       reference_path: str = "", live_path: str = "", indent: int = 2) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        json_data = self.to_json_dict(mutation_category, mutation_name, reference_path, live_path)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=indent, ensure_ascii=False)
        print(f"📄 Análisis DTW exportado a JSON: {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> 'OnsetDTWAnalysisResult':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        matches = []
        for match_data in data['matches']:
            match = OnsetMatch(
                ref_onset=match_data['ref_onset'],
                live_onset=match_data['live_onset'],
                ref_note=match_data['ref_note'],
                live_note=match_data['live_note'],
                time_adjustment=match_data['time_adjustment'],
                classification=OnsetType(match_data['classification']),
                note_similarity=match_data.get('note_similarity', 0.0),
            )
            matches.append(match)

        missing_onsets = [(onset['onset'], onset['pitch']) for onset in data['missing_onsets']]
        extra_onsets = [(onset['onset'], onset['pitch']) for onset in data['extra_onsets']]

        return cls(
            matches=matches,
            missing_onsets=missing_onsets,
            extra_onsets=extra_onsets,
        )
