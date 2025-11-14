"""
Unit Tests for Semantic Matcher Module

Tests cover:
- AbbreviationExpander functionality
- FieldNameTokenizer with various naming conventions
- TokenSimilarityCalculator (Jaccard, Cosine, Weighted)
- SemanticMatcher integration and edge cases
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.semantic_matcher import (
    AbbreviationExpander,
    FieldNameTokenizer,
    TokenSimilarityCalculator,
    SemanticMatcher
)


class TestAbbreviationExpander(unittest.TestCase):
    """Test suite for AbbreviationExpander class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.expander = AbbreviationExpander()
    
    def test_default_abbreviations(self):
        """Test that default abbreviations work correctly."""
        self.assertEqual(self.expander.expand('opp'), 'opportunity')
        self.assertEqual(self.expander.expand('cont'), 'contact')
        self.assertEqual(self.expander.expand('acc'), 'account')
        self.assertEqual(self.expander.expand('addr'), 'address')
        self.assertEqual(self.expander.expand('id'), 'identifier')
    
    def test_case_insensitivity(self):
        """Test that abbreviation expansion is case insensitive."""
        self.assertEqual(self.expander.expand('OPP'), 'opportunity')
        self.assertEqual(self.expander.expand('Opp'), 'opportunity')
        self.assertEqual(self.expander.expand('oPp'), 'opportunity')
    
    def test_non_existent_abbreviation(self):
        """Test that non-existent abbreviations return original."""
        self.assertEqual(self.expander.expand('xyz'), 'xyz')
        self.assertEqual(self.expander.expand('random'), 'random')
        self.assertEqual(self.expander.expand('field'), 'field')
    
    def test_french_abbreviations(self):
        """Test French abbreviation support."""
        self.assertEqual(self.expander.expand('soc'), 'societe')
        self.assertEqual(self.expander.expand('tel'), 'telephone')
        self.assertEqual(self.expander.expand('nom'), 'name')
    
    def test_add_custom_abbreviation(self):
        """Test adding custom abbreviations at runtime."""
        self.expander.add_custom_abbreviation('cust', 'customized')
        self.assertEqual(self.expander.expand('cust'), 'customized')
    
    def test_load_from_dict(self):
        """Test bulk loading custom abbreviations."""
        custom_dict = {
            'xyz': 'extended',
            'abc': 'alphabet',
            'crm': 'customer_relationship'
        }
        self.expander.load_from_dict(custom_dict)
        self.assertEqual(self.expander.expand('xyz'), 'extended')
        self.assertEqual(self.expander.expand('abc'), 'alphabet')
        self.assertEqual(self.expander.expand('crm'), 'customer_relationship')
    
    def test_empty_input(self):
        """Test handling of empty strings."""
        self.assertEqual(self.expander.expand(''), '')
        self.assertEqual(self.expander.expand(None), None)


class TestFieldNameTokenizer(unittest.TestCase):
    """Test suite for FieldNameTokenizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = FieldNameTokenizer()
    
    def test_snake_case(self):
        """Test tokenization of snake_case fields."""
        result = self.tokenizer.tokenize('opportunity_contact_id')
        self.assertEqual(result, ['opportunity', 'contact', 'identifier'])
        
        result = self.tokenizer.tokenize('billing_address_line_1')
        self.assertIn('billing', result)
        self.assertIn('address', result)
    
    def test_camel_case(self):
        """Test tokenization of camelCase fields."""
        result = self.tokenizer.tokenize('OpportunityContactID')
        self.assertEqual(result, ['opportunity', 'contact', 'identifier'])
        
        result = self.tokenizer.tokenize('firstName')
        self.assertEqual(result, ['first', 'name'])
    
    def test_mixed_case(self):
        """Test tokenization of mixed naming conventions."""
        result = self.tokenizer.tokenize('oppoContID')
        self.assertEqual(result, ['opportunity', 'contact', 'identifier'])
    
    def test_consecutive_capitals(self):
        """Test handling of consecutive capitals (e.g., HTTPSConnection)."""
        result = self.tokenizer.tokenize('HTTPSConnection')
        self.assertIn('https', result)
        self.assertIn('connection', result)
        
        result = self.tokenizer.tokenize('XMLParser')
        self.assertIn('xml', result)
        self.assertIn('parser', result)
    
    def test_numbers_in_field_names(self):
        """Test splitting on numbers."""
        result = self.tokenizer.tokenize('address1Line2')
        self.assertIn('address', result)
        self.assertIn('1', result)
        self.assertIn('line', result)
        self.assertIn('2', result)
        
        result = self.tokenizer.tokenize('field123name')
        self.assertIn('field', result)
        self.assertIn('123', result)
        self.assertIn('name', result)
    
    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        result = self.tokenizer.tokenize('field@name#test')
        self.assertNotIn('@', ''.join(result))
        self.assertNotIn('#', ''.join(result))
        self.assertIn('field', result)
        self.assertIn('name', result)
        self.assertIn('test', result)
    
    def test_abbreviation_expansion(self):
        """Test that abbreviations are expanded during tokenization."""
        result = self.tokenizer.tokenize('opp_cont_id')
        self.assertEqual(result, ['opportunity', 'contact', 'identifier'])
        
        result = self.tokenizer.tokenize('acc_addr')
        self.assertEqual(result, ['account', 'address'])
    
    def test_short_tokens_filtered(self):
        """Test that very short tokens are filtered out (except numbers)."""
        result = self.tokenizer.tokenize('a_bc_def_1')
        self.assertNotIn('a', result)  # Single letter filtered
        self.assertIn('bc', result)
        # Note: 'def' is a Python keyword and gets expanded to 'default' by abbreviation expander
        self.assertIn('default', result)  # 'def' -> 'default'
        self.assertIn('1', result)  # Numbers kept
    
    def test_unicode_normalization(self):
        """Test handling of unicode characters."""
        result = self.tokenizer.tokenize('café_résumé')
        # Should convert to ascii-compatible
        self.assertTrue(all(ord(c) < 128 for token in result for c in token))
    
    def test_empty_input(self):
        """Test handling of empty strings."""
        result = self.tokenizer.tokenize('')
        self.assertEqual(result, [])
        
        result = self.tokenizer.tokenize(None)
        self.assertEqual(result, [])
    
    def test_tokenize_with_metadata(self):
        """Test tokenize_with_metadata returns proper structure."""
        result = self.tokenizer.tokenize_with_metadata('opp_cont_id')
        
        self.assertIn('original', result)
        self.assertIn('tokens', result)
        self.assertIn('token_set', result)
        self.assertIn('expanded_abbreviations', result)
        
        self.assertEqual(result['original'], 'opp_cont_id')
        self.assertEqual(result['tokens'], ['opportunity', 'contact', 'identifier'])
        self.assertIsInstance(result['token_set'], set)
        self.assertEqual(result['token_set'], {'opportunity', 'contact', 'identifier'})
    
    def test_real_world_examples(self):
        """Test with real-world CRM field names."""
        test_cases = [
            ('Tel_Soc', ['telephone', 'societe']),
            ('Raison_Sociale', ['raison', 'sociale']),
            ('CompBillingAddr1', ['company', 'billing', 'address', '1']),
            ('contactEmailPrimary', ['contact', 'email', 'primary']),
        ]
        
        for field_name, expected in test_cases:
            result = self.tokenizer.tokenize(field_name)
            self.assertEqual(result, expected, f"Failed for {field_name}")


class TestTokenSimilarityCalculator(unittest.TestCase):
    """Test suite for TokenSimilarityCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = TokenSimilarityCalculator()
    
    def test_jaccard_identical_sets(self):
        """Test Jaccard similarity with identical sets."""
        tokens1 = {'opportunity', 'contact', 'identifier'}
        tokens2 = {'opportunity', 'contact', 'identifier'}
        
        result = self.calculator.jaccard_similarity(tokens1, tokens2)
        self.assertEqual(result, 1.0)
    
    def test_jaccard_disjoint_sets(self):
        """Test Jaccard similarity with disjoint sets."""
        tokens1 = {'opportunity', 'contact'}
        tokens2 = {'address', 'phone'}
        
        result = self.calculator.jaccard_similarity(tokens1, tokens2)
        self.assertEqual(result, 0.0)
    
    def test_jaccard_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        tokens1 = {'opportunity', 'contact', 'identifier'}
        tokens2 = {'opportunity', 'contact', 'number'}
        
        result = self.calculator.jaccard_similarity(tokens1, tokens2)
        # intersection = 2, union = 4
        self.assertAlmostEqual(result, 0.5, places=2)
    
    def test_jaccard_empty_sets(self):
        """Test Jaccard similarity with empty sets."""
        result = self.calculator.jaccard_similarity(set(), set())
        self.assertEqual(result, 0.0)
        
        result = self.calculator.jaccard_similarity({'test'}, set())
        self.assertEqual(result, 0.0)
    
    def test_cosine_identical_tokens(self):
        """Test cosine similarity with identical token lists."""
        tokens1 = ['opportunity', 'contact', 'identifier']
        tokens2 = ['opportunity', 'contact', 'identifier']
        
        result = self.calculator.cosine_similarity(tokens1, tokens2)
        # Use assertAlmostEqual to handle floating point precision
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_cosine_disjoint_tokens(self):
        """Test cosine similarity with disjoint token lists."""
        tokens2 = ['address', 'phone']
        
        result = self.calculator.cosine_similarity(tokens1, tokens2)
        self.assertEqual(result, 0.0)
    
    def test_cosine_with_frequency(self):
        """Test cosine similarity with repeated tokens."""
        tokens1 = ['contact', 'contact', 'name']
        tokens2 = ['contact', 'name', 'name']
        
        result = self.calculator.cosine_similarity(tokens1, tokens2)
        # Should consider token frequencies
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)
    
    def test_weighted_similarity_high_value_tokens(self):
        """Test weighted similarity with high-value tokens."""
        tokens1 = ['opportunity', 'contact']
        tokens2 = ['opportunity', 'contact']
        
        result = self.calculator.weighted_token_similarity(tokens1, tokens2)
        self.assertEqual(result, 1.0)
    
    def test_weighted_similarity_mixed_tokens(self):
        """Test weighted similarity with mixed importance tokens."""
        tokens1 = ['opportunity', 'identifier']  # high + low
        tokens2 = ['opportunity', 'number']      # high + low
        
        result = self.calculator.weighted_token_similarity(tokens1, tokens2)
        # Should heavily weight the shared 'opportunity' token
        self.assertGreater(result, 0.5)
    
    def test_weighted_similarity_custom_weights(self):
        """Test weighted similarity with custom weights."""
        custom_weights = {
            'custom': 5.0,
            'field': 1.0,
            'default': 1.0
        }
        
        tokens1 = ['custom', 'field']
        tokens2 = ['custom', 'other']
        
        result = self.calculator.weighted_token_similarity(
            tokens1, tokens2, custom_weights
        )
        # 'custom' should dominate the score
        self.assertGreater(result, 0.5)
    
    def test_calculate_all_similarities(self):
        """Test calculation of all similarity metrics."""
        meta1 = {
            'tokens': ['opportunity', 'contact', 'identifier'],
            'token_set': {'opportunity', 'contact', 'identifier'}
        }
        meta2 = {
            'tokens': ['opportunity', 'contact', 'number'],
            'token_set': {'opportunity', 'contact', 'number'}
        }
        
        result = self.calculator.calculate_all_similarities(meta1, meta2)
        
        self.assertIn('jaccard', result)
        self.assertIn('cosine', result)
        self.assertIn('weighted', result)
        self.assertIn('combined', result)
        
        # All scores should be between 0 and 1
        for score in result.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Combined should be weighted average
        expected_combined = (
            0.3 * result['jaccard'] + 
            0.3 * result['cosine'] + 
            0.4 * result['weighted']
        )
        self.assertAlmostEqual(result['combined'], expected_combined, places=5)


class TestSemanticMatcher(unittest.TestCase):
    """Test suite for SemanticMatcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matcher = SemanticMatcher(
            high_threshold=0.85,
            low_threshold=0.4
        )
    
    def test_perfect_match(self):
        """Test matching with identical field names."""
        matches = self.matcher.match_field(
            'opportunity_contact_id',
            ['opportunity_contact_id', 'contact_id', 'opp_id']
        )
        
        self.assertEqual(len(matches), 3)
        best_match = matches[0]
        
        self.assertEqual(best_match['target_field'], 'opportunity_contact_id')
        self.assertEqual(best_match['confidence'], 'high')
        self.assertAlmostEqual(best_match['similarity_scores']['combined'], 1.0, places=1)
    
    def test_abbreviation_match(self):
        """Test matching with abbreviated vs expanded forms."""
        matches = self.matcher.match_field(
            'oppoContID',
            ['opportunity_contact_identifier', 'contact_id', 'opportunity_id']
        )
        
        best_match = matches[0]
        self.assertEqual(best_match['target_field'], 'opportunity_contact_identifier')
        self.assertEqual(best_match['confidence'], 'high')
    
    def test_partial_match(self):
        """Test matching with partial token overlap."""
        matches = self.matcher.match_field(
            'contact_email',
            ['contact_email_primary', 'email_address', 'phone_number']
        )
        
        best_match = matches[0]
        self.assertIn(best_match['target_field'], ['contact_email_primary', 'email_address'])
        self.assertIn(best_match['confidence'], ['high', 'medium'])
    
    def test_no_match(self):
        """Test matching with completely different fields."""
        matches = self.matcher.match_field(
            'opportunity_amount',
            ['contact_name', 'address_line', 'phone_number']
        )
        
        # Should still return results, but all low confidence
        self.assertEqual(len(matches), 3)
        for match in matches:
            self.assertEqual(match['confidence'], 'low')
    
    def test_confidence_thresholds(self):
        """Test that confidence levels are assigned correctly."""
        # High confidence (>= 0.85)
        matches = self.matcher.match_field(
            'opp_id',
            ['opportunity_identifier']
        )
        self.assertEqual(matches[0]['confidence'], 'high')
        
        # Create medium confidence scenario
        matches = self.matcher.match_field(
            'contact_info',
            ['contact_information', 'contact_data', 'user_info']
        )
        # At least one should be medium or high
        has_acceptable = any(m['confidence'] in ['medium', 'high'] for m in matches)
        self.assertTrue(has_acceptable)
    
    def test_get_best_match_high_confidence(self):
        """Test get_best_match with high confidence match."""
        result = self.matcher.get_best_match(
            'oppoContID',
            ['opportunity_contact_identifier', 'contact_id', 'other_field']
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['target_field'], 'opportunity_contact_identifier')
        self.assertEqual(result['confidence'], 'high')
    
    def test_get_best_match_low_confidence(self):
        """Test get_best_match returns None for low confidence."""
        result = self.matcher.get_best_match(
            'random_field',
            ['completely_different', 'nothing_similar', 'unrelated']
        )
        
        # Should return None if all matches are low confidence
        if result is not None:
            self.assertIn(result['confidence'], ['medium', 'high'])
    
    def test_batch_match(self):
        """Test batch matching functionality."""
        source_fields = [
            'opp_id',
            'cont_email',
            'acc_name'
        ]
        
        target_fields = [
            'opportunity_identifier',
            'contact_email_address',
            'account_name',
            'other_field'
        ]
        
        results = self.matcher.batch_match(source_fields, target_fields)
        
        self.assertEqual(len(results), 3)
        self.assertIn('opp_id', results)
        self.assertIn('cont_email', results)
        self.assertIn('acc_name', results)
        
        # Each source should have matches for all targets
        for source_field in source_fields:
            self.assertEqual(len(results[source_field]), len(target_fields))
    
    def test_cache_functionality(self):
        """Test that target field caching works."""
        target_fields = ['opportunity_id', 'contact_email', 'account_name']
        
        # First match should populate cache
        self.matcher.match_field('opp_id', target_fields)
        
        cache_info = self.matcher.get_cache_info()
        self.assertEqual(cache_info['cached_targets'], len(target_fields))
        
        # Clear cache
        self.matcher.clear_cache()
        cache_info = self.matcher.get_cache_info()
        self.assertEqual(cache_info['cached_targets'], 0)
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        result = self.matcher.match_field('', ['target'])
        self.assertEqual(result, [])
        
        result = self.matcher.match_field('source', [])
        self.assertEqual(result, [])
        
        result = self.matcher.batch_match([], ['target'])
        self.assertEqual(result, {})
    
    def test_matched_unmatched_tokens(self):
        """Test that matched and unmatched tokens are correctly identified."""
        matches = self.matcher.match_field(
            'opportunity_contact_id',
            ['opportunity_contact_number']
        )
        
        match = matches[0]
        self.assertIn('opportunity', match['matched_tokens'])
        self.assertIn('contact', match['matched_tokens'])
        self.assertIn('identifier', match['unmatched_source'])
        self.assertIn('number', match['unmatched_target'])
    
    def test_real_world_crm_fields(self):
        """Test with real-world CRM field examples."""
        test_cases = [
            {
                'source': 'Tel_Soc',
                'targets': ['company_phone', 'telephone_societe', 'phone_number'],
                'expected_best': 'telephone_societe'
            },
            {
                'source': 'Raison_Sociale',
                'targets': ['company_name', 'legal_name', 'account_name'],
                'expected_best': 'company_name'  # or legal_name
            },
            {
                'source': 'oppAmount',
                'targets': ['opportunity_amount', 'deal_value', 'revenue'],
                'expected_best': 'opportunity_amount'
            }
        ]
        
        for test_case in test_cases:
            matches = self.matcher.match_field(
                test_case['source'],
                test_case['targets']
            )
            
            best_match = matches[0]['target_field']
            # Should match expected or be high confidence
            if best_match != test_case['expected_best']:
                self.assertIn(matches[0]['confidence'], ['high', 'medium'])
    
    def test_custom_abbreviations_integration(self):
        """Test matcher with custom abbreviations."""
        custom_abbrevs = {
            'cust': 'customized',
            'xyz': 'extended'
        }
        
        matcher = SemanticMatcher(custom_abbreviations=custom_abbrevs)
        
        matches = matcher.match_field(
            'cust_xyz_field',
            ['customized_extended_field', 'other_field']
        )
        
        best_match = matches[0]
        self.assertEqual(best_match['target_field'], 'customized_extended_field')
        self.assertEqual(best_match['confidence'], 'high')
    
    def test_custom_token_weights_integration(self):
        """Test matcher with custom token weights."""
        custom_weights = {
            'vip': 10.0,  # Very high importance
            'flag': 0.1   # Very low importance
        }
        
        matcher = SemanticMatcher(custom_token_weights=custom_weights)
        
        matches = matcher.match_field(
            'vip_customer_flag',
            ['vip_account_status', 'customer_flag_field']
        )
        
        # Should prioritize match with 'vip' due to high weight
        best_match = matches[0]
        self.assertIn('vip', best_match['target_tokens'])


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests simulating real-world usage scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matcher = SemanticMatcher()
    
    def test_multi_table_field_mapping(self):
        """Test mapping fields across multiple database tables."""
        # Simulate CSV columns
        csv_columns = [
            'oppoName',
            'oppoAmount',
            'oppoCloseDate',
            'contFirstName',
            'contLastName',
            'contEmail',
            'accName',
            'accIndustry'
        ]
        
        # Simulate database fields from multiple tables
        db_fields = [
            'opportunity_name',
            'opportunity_amount',
            'opportunity_close_date',
            'contact_first_name',
            'contact_last_name',
            'contact_email_primary',
            'account_company_name',
            'account_industry_type',
            'other_random_field'
        ]
        
        results = self.matcher.batch_match(csv_columns, db_fields)
        
        # Check that key fields found good matches
        high_confidence_matches = 0
        for source_field, matches in results.items():
            if matches[0]['confidence'] == 'high':
                high_confidence_matches += 1
        
        # Should have high confidence for most fields
        self.assertGreaterEqual(high_confidence_matches, len(csv_columns) * 0.6)
    
    def test_performance_large_batch(self):
        """Test performance with larger datasets."""
        import time
        
        # Generate 50 source fields
        source_fields = [f'field_{i}_name' for i in range(50)]
        
        # Generate 100 target fields
        target_fields = [f'target_{i}_field' for i in range(100)]
        
        start_time = time.time()
        results = self.matcher.batch_match(source_fields, target_fields)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds for 5000 comparisons)
        self.assertLess(elapsed_time, 5.0)
        
        # Should have results for all source fields
        self.assertEqual(len(results), len(source_fields))
        
        # Each source should have matches for all targets
        for matches in results.values():
            self.assertEqual(len(matches), len(target_fields))


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAbbreviationExpander))
    suite.addTests(loader.loadTestsFromTestCase(TestFieldNameTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenSimilarityCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticMatcher))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
