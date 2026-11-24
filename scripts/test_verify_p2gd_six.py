import unittest
from verify_p2gd_six import verify_prefix_state


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.old=[dict(iteration=t,weight_sha256=str(t)) for t in range(253,269)]
        self.new=self.old+[dict(iteration=t,weight_sha256=str(t)) for t in range(269,301)]

    def test_later_audits_are_not_initial_state_mismatches(self):
        self.assertEqual(verify_prefix_state(self.new,self.old),16)

    def test_changed_common_state_fails(self):
        changed=[dict(r) for r in self.new]
        changed[5]['weight_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'differs'):
            verify_prefix_state(changed,self.old)

    def test_missing_iteration_fails(self):
        with self.assertRaisesRegex(ValueError,'Missing'):
            verify_prefix_state(self.new[1:],self.old)

    def test_empty_records_do_not_pass(self):
        with self.assertRaises(ValueError):
            verify_prefix_state([],[])


if __name__=='__main__':
    unittest.main()
