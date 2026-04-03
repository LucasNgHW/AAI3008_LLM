import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestMaterialDeletion(unittest.TestCase):

    def test_delete_material_keeps_db_row_when_qdrant_fails(self):
        import pipeline.material_ingestion as mi

        material = {"id": 3, "filename": "lecture3.pdf", "content": b"x"}
        with patch.object(mi, "get_material", return_value=material), \
             patch.object(mi, "delete_material_chunks", side_effect=RuntimeError("boom")), \
             patch.object(mi, "delete_material") as mock_delete_db:
            result = mi.delete_material_everywhere(3)

        self.assertFalse(result["deleted"])
        self.assertEqual(result["reason"], "qdrant_error")
        mock_delete_db.assert_not_called()

    def test_delete_all_keeps_db_rows_when_collection_verify_fails(self):
        import pipeline.material_ingestion as mi

        mock_client = MagicMock()
        with patch.object(mi, "collection_exists", side_effect=[True, True]), \
             patch.object(mi, "get_client", return_value=mock_client), \
             patch.object(mi, "delete_all_materials") as mock_delete_db:
            result = mi.delete_all_materials_everywhere()

        self.assertFalse(result["deleted"])
        self.assertEqual(result["reason"], "qdrant_verify_failed")
        mock_delete_db.assert_not_called()


if __name__ == "__main__":
    unittest.main()
