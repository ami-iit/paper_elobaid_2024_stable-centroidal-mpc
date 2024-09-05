
classdef Step

    properties
        position
        init_time {mustBeNumeric, mustBeNonnegative}
        end_time {mustBeNumeric, mustBeNonnegative}
        rotation
    end

    methods
        function obj = Step(position, rpy_rad, init_time, end_time)
            if (length(position) ~= 3)
                error ('The position should be a vector with a length of 3')
            else
                obj.position = position;
            end

            rpy = rpy_rad * 180 / pi;

            obj.rotation = rotz(rpy(3)) * roty(rpy(2)) * rotx(rpy(1));
            obj.init_time = init_time;
            obj.end_time = end_time;
        end


    end
end
