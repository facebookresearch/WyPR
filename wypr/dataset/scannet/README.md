### Prepare ScanNet Data

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). 
   - Move/link the `scans` folder such that under `scans` there should be folders with names such as `scene0001_01`.

2. Move/link the cgal output to `cgal_output`. 
   - To detect planes using cgal, check [RUNNING](../../../docs/RUNNING.md).
   - Alternatively, you can change the path (`CGAL_DIR`) in `preprocess_scannet_all_points.py`

3. Extract point clouds and annotations (semantic seg, instance seg etc.) by running `python preprocess_scannet_all_points.py`, which will create a folder named `scannet_all_points` here.

4. Move/link the proposals files to `proposals`.
   - To generate proposals, check []().