import noise
import numpy as np
import trimesh

GRASS_COLOR = [34, 139, 34, 255]
DIRT_COLOR = [139, 69, 19, 255]
WATER_COLOR = [0, 0, 255, 255]


def create_large_plane(
    size=100,
    resolution=1,
    terrain_scale=20.0,
    terrain_octaves=6,
    terrain_persistence=0.5,
    terrain_lacunarity=2.0,
    terrain_height=30.0,
    hole_scale=60.0,
    hole_octaves=2,
    hole_threshold=-0.4,
    hole_depth=-40.0,
):
    print("Creating large plane...")
    num = int(size / resolution)
    x = np.linspace(-size / 2, size / 2, num + 1)
    y = np.linspace(-size / 2, size / 2, num + 1)
    xx, yy = np.meshgrid(x, y)

    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            tx = xx[i, j] / terrain_scale
            ty = yy[i, j] / terrain_scale
            terrain_val = noise.pnoise2(
                tx,
                ty,
                octaves=terrain_octaves,
                persistence=terrain_persistence,
                lacunarity=terrain_lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=100,
            )

            hx = xx[i, j] / hole_scale
            hy = yy[i, j] / hole_scale
            hole_val = noise.pnoise2(
                hx,
                hy,
                octaves=hole_octaves,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=1024,
                repeaty=1024,
                base=200,
            )

            if hole_val < hole_threshold:
                z = hole_depth
            else:
                z = terrain_val * terrain_height

            zz[i, j] = z

    print(f"Height stats: min={zz.min():.2f}, max={zz.max():.2f}")

    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    faces = []
    for i in range(num):
        for j in range(num):
            idx = i * (num + 1) + j
            idx_r = idx + 1
            idx_d = idx + (num + 1)
            idx_dr = idx_d + 1
            faces.append([idx, idx_r, idx_dr])
            faces.append([idx, idx_dr, idx_d])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))

    face_colors = []
    for f in mesh.faces:
        centroid = mesh.vertices[f].mean(axis=0)
        val = noise.pnoise2(
            centroid[0] / terrain_scale,
            centroid[1] / terrain_scale,
            octaves=2,
            persistence=0.5,
            lacunarity=2.0,
            repeatx=1024,
            repeaty=1024,
            base=150,
        )
        face_colors.append(GRASS_COLOR if val > 0 else DIRT_COLOR)

    mesh.visual.face_colors = np.array(face_colors)

    print("Mesh created with colors.")
    return mesh


def create_water_plane(
    size=120, resolution=1, wave_scale=10.0, wave_height=3.0, wave_freq=0.2
):
    print("Creating water plane with sine waves...")
    num = int(size / resolution)
    x = np.linspace(-size / 2, size / 2, num + 1)
    y = np.linspace(-size / 2, size / 2, num + 1)
    xx, yy = np.meshgrid(x, y)

    zz = wave_height * np.sin(wave_freq * xx) * np.cos(wave_freq * yy)

    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    faces = []
    for i in range(num):
        for j in range(num):
            idx = i * (num + 1) + j
            idx_r = idx + 1
            idx_d = idx + (num + 1)
            idx_dr = idx_d + 1
            faces.append([idx, idx_r, idx_dr])
            faces.append([idx, idx_dr, idx_d])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.visual.face_colors = np.array([WATER_COLOR] * len(mesh.faces))
    print("Water plane created.")
    return mesh


def mean_curvature_flow(mesh, iters=100, tau=0.1):
    print("Starting mean curvature flow smoothing...")
    adj = mesh.vertex_neighbors
    z = mesh.vertices[:, 2].copy()
    for k in range(iters):
        z_new = np.empty_like(z)
        for i, nbrs in enumerate(adj):
            z_new[i] = z[i] + tau * (np.mean(z[nbrs]) - z[i]) if nbrs else z[i]
        z = z_new
        if k % 10 == 0 or k == iters - 1:
            print(
                f"""Iteration {k+1}/{iters},
                    mean z: {np.mean(z):.4f},
                    max z: {np.max(z):.4f},
                    min z: {np.min(z):.4f}"""
            )
    mesh.vertices[:, 2] = z
    print("Mean curvature flow complete.")


if __name__ == "__main__":
    terrain = create_large_plane()
    water = create_water_plane()
    scene = trimesh.Scene()
    scene.add_geometry(water)
    scene.add_geometry(terrain)
    scene.set_camera(
        angles=[np.deg2rad(60), 0, np.deg2rad(45)], distance=250, center=[0, 0, 0]
    )
    print("Showing initial scene...")
    scene.show()

    mean_curvature_flow(terrain, iters=200, tau=0.05)

    scene2 = trimesh.Scene()
    scene2.add_geometry(water)
    scene2.add_geometry(terrain)
    scene2.set_camera(
        angles=[np.deg2rad(60), 0, np.deg2rad(45)], distance=250, center=[0, 0, 0]
    )
    print("Showing smoothed scene...")
    scene2.show()
