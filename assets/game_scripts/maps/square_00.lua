---
--- Created by ruben.
--- DateTime: 30/11/17 18:58
---
local map = {}

--[[
    Description of entity objects:
    * --> wall
    A --> apple +1
    F --> fungi -2
    G --> goal +10
    H --> horizontal door
    I --> vertical door
    L --> lemon -1
    P --> player
    S --> strawberry +2
--]]

map.entity = [[
******
*G   *
*    *
*    *
*   P*
******
]]

--[[
    Define variation in texture to get different zones:
    'A' - 'Z' --> different zones
    '.' or ' ' --> no variation in texture
--]]

map.variation = [[
......
.AAAA.
.AAAA.
.AAAA.
.AAAA.
......
]]

--[[
    No variation
--]]

map.variation = {}

return map