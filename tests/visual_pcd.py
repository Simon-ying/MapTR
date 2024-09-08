import open3d as o3d
import argparse

parser = argparse.ArgumentParser(description="Visualize a PCD file using Open3D.")
parser.add_argument("pcd_file", type=str, help="Path to the PCD file to visualize.")
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.pcd_file)

print(f"Visualizing PCD file: {args.pcd_file}")
o3d.visualization.draw_geometries([pcd])