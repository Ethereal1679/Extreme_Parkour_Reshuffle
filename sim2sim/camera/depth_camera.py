import mujoco

model = mujoco.MjModel.from_xml_path("./resources/panda_with_camera/depth_camera_scene.xml")
sim = mujoco.MjSim(model)

## a is a tuple if depth is True and a numpy array if depth is False ##
a = sim.render(width=200, height=200, camera_name='rgb', depth=True)
rgb_img = a[0]
depth_img = a[1]