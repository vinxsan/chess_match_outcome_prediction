'''
Use this script to test inference function 
'''
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from inference import build_features, predict_match_outcome

# define a valide input match
valid_match = {
    'white_rating': 1600, 'black_rating': 1450, 'increment_code': '10+0',
    'opening_eco': 'B20', 'opening_ply': 4, 'rated': True
}

PROBA_COLS = ['proba_black', 'proba_draw', 'proba_white']

# test unknown opening warn
def test_unknown_opening_warns():
    bad_match = {**valid_match, 'opening_eco': 'ZZZ'}
    with pytest.warns(UserWarning, match = 'opening_eco'):
         result = predict_match_outcome(bad_match)
    assert result['opening_eco_unknown'].iloc[0]

# test required colum error
def test_required_columns_raises():
     bad_match = {k: v for k,v in valid_match.items() if k != 'opening_eco'}
     with pytest.raises(ValueError, match = 'opening_eco'):
          predict_match_outcome(bad_match)

# test differnt name using
def test_both_opening_column_names_give_same_result():
    with_ply = predict_match_outcome(valid_match)
    moves_match = {k: v for k, v in valid_match.items() if k != 'opening_ply'}
    moves_match['opening_moves'] = valid_match['opening_ply']
    with_moves = predict_match_outcome(moves_match)
    pd.testing.assert_frame_equal(with_ply, with_moves)

# test build_features on modified data input

def test_build_features_keeps_rows_and_does_not_modify_input():
    raw = pd.DataFrame([valid_match, {**valid_match, 'increment_code': '1+0'}], index=[10, 20])
    raw_before = raw.copy()                             

    features = build_features(raw)

    assert len(features) == len(raw)                     
    assert list(features.index) == [10, 20]              
    pd.testing.assert_frame_equal(raw, raw_before)       
    assert list(features['match_category']) == ['Rapid', 'Bullet']


# test if probabilities' sum == 1
def test_probabilities_sum_to_one():
    result = predict_match_outcome([valid_match, {**valid_match, 'white_rating': 900}])
    proba_cols = ['proba_black', 'proba_draw', 'proba_white']
    assert len(result) == 2
    assert np.allclose(result[proba_cols].sum(axis=1), 1)

# test increment code format
def test_malformed_increment_code_raises():
    bad_match = {**valid_match, 'increment_code': '10'}
    with pytest.raises(ValueError, match='increment_code'):
        predict_match_outcome(bad_match)

# compare with expected prediction
def test_expected_prediction():
    expected = pd.read_csv(Path(__file__).parent / 'expected_predictions.csv')
    matches = [valid_match, 
                      {'white_rating': 600, 'black_rating': 1450, 'increment_code': '10+10',
                       'opening_eco': 'AOO', 'opening_ply': 10, 'rated': False}]
     
    result = predict_match_outcome(matches)

    numeric_cols = PROBA_COLS + ['surprise_probability']
    np.testing.assert_allclose(result[numeric_cols].to_numpy(), 
                               expected[numeric_cols].to_numpy(), rtol= 1e-6)

    flag_cols = ['flagged_as_potential_surprise', 'opening_eco_unknown']
    assert (result[flag_cols].to_numpy() == expected[flag_cols].to_numpy()).all()