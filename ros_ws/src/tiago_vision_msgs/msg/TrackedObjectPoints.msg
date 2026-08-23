# One object recognised by the vision pipeline. `points` samples the
# object's visible surface at up to 3 spread-out locations (rather than a
# single centroid), so a downstream node can fit a circle through them to
# recover a cylindrical object's true central axis/position for grasp
# planning, or use them as a lightweight footprint for collision checks.

string class_name                # YOLO class name, e.g. 'coke can'
string shape                     # geometric shape, e.g. 'cylinder' (see OBJECT_SHAPES in the vision node)
geometry_msgs/Point[<=3] points  # up to 3 points on the object's visible surface, camera optical frame