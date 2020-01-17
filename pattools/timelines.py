from pacs import Series

class ScorecardElement:
    description = None
    points = 0
    def __init__(self, description, points):
        self.description = description
        self.points = points

class Filter:
    name = None
    scorecard = [] # For matching description
    min_rows = 100
    min_cols = 100
    threshold = 0

    def __init__(self, name):
        self.name = name

    def filter(study):
        series = study.find_series()

def flair_filter():
    filter = Filter('FLAIR')
    filter.scorecard.append(ScorecardElement('flair'), 100))
    filter.scorecard.append(ScorecardElement('sag'), 20))
    filter.scorecard.append(ScorecardElement('_cor'), -5))
    filter.scorecard.append(ScorecardElement('_tra'), -5))
    filter.scorecard.append(ScorecardElement('_cor'), -5))
    filter.scorecard.append(ScorecardElement('t2'), 45))
    filter.scorecard.append(ScorecardElement('t1'), -45))
    filter.scorecard.append(ScorecardElement('3d'), 25))
    filter.threshold = 100

def mprage_filter():
    filter = Filter('MPRAGE')
    filter.scorecard.append(ScorecardElement('mprage'), 100))
    filter.scorecard.append(ScorecardElement('sag'), 20))
    filter.scorecard.append(ScorecardElement('_cor'), -5))
    filter.scorecard.append(ScorecardElement('_tra'), -5))
    filter.scorecard.append(ScorecardElement('_cor'), -5))
    filter.scorecard.append(ScorecardElement('t1'), 30))
    filter.scorecard.append(ScorecardElement('t2'), -30))
    filter.scorecard.append(ScorecardElement('3d'), 25))
    filter.scorecard.append(ScorecardElement('c+'), -5))
    filter.threshold = 100


class Timeline:
    patient_id = None
    path = None
    start_date = None
    end_date = None

    brain_mask = None

    def __init__(self, path, patient_id=None):
        # Try to load from path...

        # If that doesn't work, try to create from PACS
        series_list.sort(key=lambda s: s.study_date)
        self.series_list = series_list
        if len(series_list)>0:
            self.start_date = series_list[0].study_date
            self.end_date = series_list[-1].study_date
