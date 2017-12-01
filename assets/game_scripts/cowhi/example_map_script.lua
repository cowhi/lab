local make_map = require 'cowhi.example_map_generator'
local pickups = require 'cowhi.example_map_objects'
local rand_gen = require 'common.random'
local api = {}
local map_name = 'square_00'


function set_spawn(map_str)
  string.gsub(map_str, "P"," ")
  count = string.len(map_str)
  repeat
    pos = rand_gen.uniformInt(1, count)
    print("Selected spawn position", pos)
  until(map_str:sub(pos, pos) == ' ')
  return map_str:sub(1, pos-1) .. 'P' .. map_str:sub(pos+1)
end

function api:start(episode, seed)
  make_map.seedRng(seed)
  api._count = 0
end

function api:commandLine(oldCommandLine)
  return make_map.commandLine(oldCommandLine)
end

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:nextMap()
  map_data = require('maps.'..map_name)
  map_data.entity = set_spawn(map_data.entity)
  return make_map.makeMap(map_name, map_data.entity, map_data.variation)
end

return api