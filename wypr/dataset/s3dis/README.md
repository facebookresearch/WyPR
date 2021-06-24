### Prepare S3DIS

1. Please download Stanford3dDataset_v1.2_Aligned_Version.zip from https://goo.gl/forms/4SoGp4KtH1jfRqEj2
   - Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations/ceiling_1.txt has wrong charcter at line 180389. Please fix it manually.

2. Pre-process data
   - Run `convert_pc2npy` in `process.py` to generate point cloud file from raw data.
   - Run `convert2xyzn` in `process.py` to convert raw data (*.npy) to *.xyzn for plane detection.
   - Run `run_shape_det` in `process.py` to detect planes using cgal. To install cgal, follow [RUNNING.md](../../../docs/RUNNING.md).
   - Then `fix_unassigned_points` in `process.py` to post-process cgal results.

3. Extract point clouds and annotations (semantic seg, bbox seg etc.) by running `python prepare_all_points.py`, which will save to `s3dis_all_points`.
