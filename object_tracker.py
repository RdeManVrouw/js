import rospy 

class ObjectTracker(object):

  def __init__(self):
    self.lane_width = 0.5 # The width of a lane (from line to line) in meters
    self.lane_occupied = {}
    self.lane_occupied["left_lane"] = False
    self.lane_occupied["right_lane"] = False
    self.lane_occupied["front"] = False

  def check_lanes(self, clusters):
    self.lane_occupied["left_lane"] = False
    self.lane_occupied["right_lane"] = False
    self.lane_occupied["front"] = False
    for c in clusters:
      #print(min(c, key = lambda p: p[0]))
      if min(c, key = lambda p: p[0])[0] <= 2.5:
        sides = ["left_lane", "front", "right_lane"]
        bounds = [[0.75, 0.25], [0.25, -0.25], [-0.25, -0.75]]
        for i in range(3):
          if c[0][1] <= bounds[i][0] and c[0][1] >= bounds[i][1]:
            self.lane_occupied[sides[i]] = True
          if c[-1][1] <= bounds[i][0] and c[-1][1] >= bounds[i][1]:
            self.lane_occupied[sides[i]] = True
          if c[0][1] <= bounds[i][1] and c[-1][1] >= bounds[i][1]:
            self.lane_occupied[sides[i]] = True
          if c[0][1] <= bounds[i][0] and c[-1][1] >= bounds[i][0]:
            self.lane_occupied[sides[i]] = True