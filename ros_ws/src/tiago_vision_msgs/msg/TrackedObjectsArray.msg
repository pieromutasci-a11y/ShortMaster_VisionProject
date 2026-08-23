# All objects tracked in one RGB-D frame, published together so a consumer
# (e.g. the grasp-planning service) has the target and the surrounding
# obstacles from the same instant. The target object is always the first
# entry when present.

std_msgs/Header header
TrackedObjectPoints[] objects