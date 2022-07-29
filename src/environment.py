#############################
# state<
#############################

# Not sure yet what the error models are going to be

# workspace is an interval [lo, hi]
# robot_loc is a float in [lo, hi]
#     Robot is effectively a point
# fov is a float

# objects is list
# each object has location and size attributes

class ODKState(State):
    def __init__(self, robot_loc, objects):

        State.__init__(self)
        self.robot_loc = robot_loc
        self.objects = objects
        self.grasped = 'none'

    # For each object that overlaps the fov by more than a tiny amount, return
    # - color
    # - lower edge and whether it is an occlusion boundary
    # - upper edge and whether it is an occlusion boundary
    def generate_observation(self):
        def generate_obj_obs(o, overlap):
            return FrozenDict(
                  {'color' : o.color,
                   'lo' : interval_lo(overlap),
                   'lo_occluded' :
                      interval_lo(o.range()) < interval_lo(obs_field)- 1e-10,
                   'hi' : interval_hi(overlap),
                   'hi_occluded' :
                       interval_hi(o.range()) > interval_hi(obs_field) + 1e-10})

        view_field = (self.robot_loc, FOV/2)
        obs_field = interval_intersection(WORKSPACE, view_field)
        obs = set()
        for o in self.objects:
            if o.pos is None: continue  # Object is being held
            overlap = interval_intersection(o.range(), obs_field)
            if interval_size(overlap) < MIN_OBJECT_DETECT_SIZE: continue
            obs.add(generate_obj_obs(o, overlap))
            # Get objects to the left and right of me and make extra detections
            ool, oor =  self.touching_objects(o)
            if ool and ool.color == o.color:rgba(255, 255, 255, 0.00); color: rgb(0, 0, 0); font-family: Calibri; font-style: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: auto; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: auto; word-spacing: 0px; -webkit-text-size-adjust: auto; -webkit-text-stroke-width: 0px; background-color: rgb(255, 255, 255); text-decoration: none;">                
                l_overlap = interval_intersection(interval_union(ool.range(), o.range()),
                                                  obs_field)
                if interval_size(l_overlap) >= MIN_OBJECT_DETECT_SIZE:
                    obs.add(generate_obj_obs(o, l_overlap))
            if oor and oor.color == o.color:rgba(255, 255, 255, 0.00); color: rgb(0, 0, 0); font-family: Calibri; font-style: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: auto; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: auto; word-spacing: 0px; -webkit-text-size-adjust: auto; -webkit-text-stroke-width: 0px; background-color: rgb(255, 255, 255); text-decoration: none;">                r_overlap = interval_intersection(interval_union(o.range(),
                                                               oor.range()),
                                                 obs_field)
                if interval_size(r_overlap) >= MIN_OBJECT_DETECT_SIZE:
                    obs.add(generate_obj_obs(o, r_overlap))
            if random.random() < OBJ_FRAGMENT_PROB:
                # Add two new detections corresponding to this object
                # Could also consider removing the original detection, but not doing that for now
                p = random.uniform(interval_lo(overlap), interval_hi(overlap))
                obs.add(generate_obj_obs(o,
                            lohi_to_interval(interval_lo(overlap), p)))
                obs.add(generate_obj_obs(o,
                            lohi_to_interval(p, interval_hi(overlap))))
        return obs

    # If there are objects abutting o, return them
    # Returns (object or None) for left and right
    # Should keep the objects sorted
    def touching_objects(self, o):
        left = None
        right = None
        for to in self.objects:
            if to == o: continue
            if within(o.lo(), to.hi(), 1e-10):
                assert left is None
                left = to
            if within(o.hi(), to.lo(), 1e-10):
                assert right is None
                right = to
        return left, right

    def execute(self, act):
        print('\nExecuting', act)
        if act == 'look' or \
                act.name in ('look_obj_hi', 'look_obj_lo', 'look_region'):
            # No arguments
            obs = self.generate_observation()
            print('Looking from', self.robot_loc, 'got obs', obs)
        elif act.name == 'move':
            (start_loc, target_loc) = act.args
            assert self.robot_loc == start_loc
            self.robot_loc = target_loc
            print('Moved to', target_loc)
            obs = None
        elif act.name == 'pick':
            # No arguments
            if self.grasped != 'none':
                # If we are already holding something, then fail
                obs = 'fail'
            else:
                # If we are within eps of the center of an object, then we pick otherwise we leave it where it is, or somewhat perturbed
                # Assume there are not two such objects!
                obs = 'fail'
                for o in self.objects:
                    if within(o.pos, self.robot_loc, PICK_TOLERANCE):
                        self.grasped = o
                        o.pos = None
                        obs = 'succeed'
                if obs == 'fail':
                    print('Attempted to pick but not near center of any obj')

        elif act.name == 'place':
            # No arguments
            # If there is free space under it, then it is placed whp, else it either stays in the gripper or slides sideways into a spot
            if self.grasped == 'none':
                # If we are not holding something, then fail
                obs = 'fail'
                print('Attempted to place, but not holding anything.')
            elif not self.collision_free((self.robot_loc, self.grasped.size)):
                obs = 'fail'
                print('Attempted to place, but space not clear')
            else:
                # Randomize this a bit
                self.grasped.pos = self.robot_loc
                self.grasped = 'none'
                obs = 'succeed'
        elif act.name in ('push_from_top', 'push_and_look1', 'push_and_look2'):
            p1 = act.args[1]
            p2 = act.args[2]
            self.robot_loc = self.robot_loc + (p2 - p1)
            touched_o = self.obj_at(self.robot_loc)
            if touched_o is None:
                # If we are not touching something, then fail
                obs = 'fail'
                print('Attempted to push, but not touching anything.')
            else:
                self.push_obj(touched_o, p2-p1)
                obs = 'succeed'
            if act.name in ('push_and_look1', 'push_and_look2'):
                return self.execute('look')
        else:
            raise NotImplementedError
        if obs == 'fail':
            tr('domain_execution_failure', act)
        self.draw()
        return obs

    # move this object, plus others it might push as well
    def push_obj(self, obj, delta):
        o_hi = obj.hi()
        o_lo = obj.lo()
        if delta > 0:
            # pushing to the right
            danger_zone = lohi_to_interval(o_hi, o_hi + delta)
            # get leftmost object that overlaps the danger zone
            pushed_o = self.overlapper(obj, danger_zone, 'leftmost')
            if pushed_o:
                self.push_obj(pushed_o, interval_hi(danger_zone) -pushed_o.lo())
        else:
            # pushing to the left
            danger_zone = lohi_to_interval(o_lo + delta, o_lo)
            # get rightmost object that overlaps the danger zone
            pushed_o = self.overlapper(obj, danger_zone, 'rightmost')
            if pushed_o:
                self.push_obj(pushed_o, interval_lo(danger_zone) -pushed_o.hi())
        # Move the actual object
        obj.pos = obj.pos + delta

    # So many ways to be smarter:  keep objects sorted, stop sooner
    def overlapper(self, orig, zone, mode):
        assert mode in ('leftmost', 'rightmost')
        for o in sorted((x for x in self.objects if x.pos is not None),
                        key = lambda x: x.pos,
                        reverse = mode == 'rightmost'):
            if o != orig and overlaps_interval(zone, o.range()):
                return o
        return None

    def obj_at(self, pos):
        for o in self.objects:
            if point_in_interval(pos, o.range()):
                return o
        return None

    def collision_free(self, reg):
        return all(not overlaps_interval(reg, o.range())
                     for o in self.objects if o.pos is not None)

    def draw(self):
        print('\n----- Current World State -----')
        print('Robot loc:', self.robot_loc)
        print('Grasped:', self.grasped)
        for o in self.objects:
            print('    ', o)
        print('-------------------------------\n')

    def pretty_string(self, _eq = True):
        return 'OneDKitchenState()'

class ObjectState:
    def __init__(self, arg_dict):
        self.__dict__.update(arg_dict)
    def range(self):
        return [self.pos, self.size]
    def lo(self):
        return self.pos - self.size
    def hi(self):
        return self.pos + self.size
    def __str__(self):
        return pretty_string(self.__dict__, False)
    __repr__ = __str__


def world_gen(num_objs, size_dist,
                colors = ('red', 'green', 'blue'),
                color_required = {},
                workspace = lohi_to_interval(0, 20.0),
                abut = False):

    # Find a position for this object in this interval
    def get_placement(size, interval):
        min_pos = interval_lo(interval) + size
        max_pos = interval_hi(interval) - size
        if abut:
            return random.sample([min_pos, max_pos])
        else:
            return random.uniform(min_pos, max_pos)

    # Do bookkeeping to remove this interval and add replacement(s)
    def update_free_intervals(interval, pos, size):
        free_intervals.remove(interval)
        obj_int = (pos, size)
        abuts_lo = within(interval_lo(interval), interval_lo(obj_int), .001)
        abuts_hi = within(interval_hi(interval), interval_hi(obj_int), .001)
        if not abuts_hi:
            free_intervals.append(
                lohi_to_interval(interval_hi(obj_int), interval_hi(interval)))
        if not abuts_lo:
            free_intervals.append(
                lohi_to_interval(interval_lo(interval), interval_lo(obj_int)))

    # Place an object of this color
    def place(c):
        size = size_dist.draw()
        random.shuffle(free_intervals)
        for interval in free_intervals:
            if size <= interval_size(interval)/2:
                pos = get_placement(size, interval)
                objects.append(ObjectState({'pos' : pos, 'size' : size,
                                            'color' : color}))
                update_free_intervals(interval, pos, size)
                return True
        return False

    # Beginning of main procedure body
    objects = []
    free_intervals = [workspace]
    m = 0
    # First, do the required ones.  For now, can only require colors
    for color, n in color_required.items():
        for j in range(n):
            m += 1
            if not place(color):
                print(f'Failed to place {j}th {color} object')
                return
    # Now do the rest
    for j in range(num_objs - m):
        color = random.choice(colors)
        win = place(color)
        if not win:
            print(f'Failed to place {j}th generic object')
            return
    return objects