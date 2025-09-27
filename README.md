# Dungeon Generator

### Assets
- Add a parchment texture at: `./images/textures/parchment.jpg`
- Recommended size: 1920x1080 (it will be auto-scaled).
- If the file is missing, the generator will use a beige background instead.

### Icons
Optional icons can be placed in `./images/icons/`:
- chest.png → treasure chests
- trap.png → traps
- torch.png → torches
- monster.png → monsters
If missing, the dungeon will still render without errors.

### Export
After generating a dungeon, the current view is automatically saved as a PNG file
in the project root directory:
- `dungeon_parchment.png` (default), or
- `dungeon_parchment_YYYYMMDD_HHMMSS.png` (if timestamping is enabled).
Timestamped exports are enabled by default. Set `USE_TIMESTAMP_EXPORT = False`
in `DungeonGenerator.py` to always overwrite the same file name.
