class ScorecardElement:
    '''The scorecard element is used to create a series filter.'''
    description = None
    points = 0
    def __init__(self, description, points):
        self.description = description
        self.points = points

class Filter:
    '''The filter class extracts a 0 or 1 series from a study (which math the filter, obvs.)'''
    name = None
    scorecard = None # For matching description
    min_rows = 100
    min_cols = 100
    threshold = 0

    def __init__(self, name):
        self.name = name
        self.scorecard = []

    def filter(self, study):
        '''Returns the series which best matches the filter, or None if there are no good matches.'''
        series = study.find_series()
        candidates = []

        for seri in series:
            score = 0
            # If the description contains a string on the scorecard..
            for se in self.scorecard:
                if se.description.lower() in seri.description.lower():
                    score += se.points #... add to the score
            if score >= self.threshold:
                candidates.append((score, seri))
        # Sort the candidates by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        # If theres a candidate, append the best
        if len(candidates) > 0:
            return candidates[0][1]
        # No valid series found
        return None


def flair_filter():
    '''Default filter for FLAIR studies'''
    filter = Filter('FLAIR')
    filter.scorecard.append(ScorecardElement('flair', 100))
    filter.scorecard.append(ScorecardElement('mprage', -100))
    filter.scorecard.append(ScorecardElement('localiser', -50))
    filter.scorecard.append(ScorecardElement('sag', 20))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('_tra', -5))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('t2', 45))
    filter.scorecard.append(ScorecardElement('t1', -45))
    filter.scorecard.append(ScorecardElement('3d', 25))
    filter.scorecard.append(ScorecardElement('dis3d', -25))
    filter.scorecard.append(ScorecardElement('report', -50))
    filter.scorecard.append(ScorecardElement('fused', -35))
    filter.scorecard.append(ScorecardElement('spine', -255))
    
    filter.threshold = 100
    return filter

def mprage_cplus_filter():
    '''Default filter for MPRAGE c+ studies'''
    filter = Filter('MPRAGE_C+')
    filter.scorecard.append(ScorecardElement('mprage', 60))
    filter.scorecard.append(ScorecardElement('flair', -100))
    filter.scorecard.append(ScorecardElement('localiser', -50))
    filter.scorecard.append(ScorecardElement('sag', 20))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('_tra', -5))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('t1', 30))
    filter.scorecard.append(ScorecardElement('t2', -30))
    filter.scorecard.append(ScorecardElement('3d', 25))
    filter.scorecard.append(ScorecardElement('dis3d', -25))
    filter.scorecard.append(ScorecardElement('c+', +50))
    filter.scorecard.append(ScorecardElement('c-', -50))
    filter.scorecard.append(ScorecardElement('report', -50))
    filter.scorecard.append(ScorecardElement('fused', -35))
    filter.scorecard.append(ScorecardElement('tv', -35))
    filter.scorecard.append(ScorecardElement('cv', -35))
    filter.scorecard.append(ScorecardElement('lv', -35))
    filter.scorecard.append(ScorecardElement('devmap_rgb', -35))
    filter.scorecard.append(ScorecardElement('spine', -255))
    filter.threshold = 100
    return filter

def mprage_cminus_filter():
    '''Default filter for MPRAGE C- studies'''
    filter = Filter('MPRAGE_C-')
    filter.scorecard.append(ScorecardElement('mprage', 100))
    filter.scorecard.append(ScorecardElement('flair', -100))
    filter.scorecard.append(ScorecardElement('localiser', -50))
    filter.scorecard.append(ScorecardElement('sag', 20))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('_tra', -5))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('t1', 30))
    filter.scorecard.append(ScorecardElement('t2', -30))
    filter.scorecard.append(ScorecardElement('3d', 25))
    filter.scorecard.append(ScorecardElement('dis3d', -25))
    filter.scorecard.append(ScorecardElement('c+', -50))
    filter.scorecard.append(ScorecardElement('c-', +50))
    filter.scorecard.append(ScorecardElement('report', -50))
    filter.scorecard.append(ScorecardElement('fused', -35))
    filter.scorecard.append(ScorecardElement('tv', -35))
    filter.scorecard.append(ScorecardElement('cv', -35))
    filter.scorecard.append(ScorecardElement('lv', -35))
    filter.scorecard.append(ScorecardElement('devmap_rgb', -35))
    filter.scorecard.append(ScorecardElement('spine', -255))
    filter.threshold = 100
    return filter

def default_filters():
    '''Default filter list'''
    return [flair_filter(), mprage_cplus_filter(), mprage_cminus_filter()]