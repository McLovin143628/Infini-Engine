//! Tilemap paint projection (P8.2b).
//!
//! Projects a selected entity's [`inf_ecs::components::Tilemap`] into the
//! [`TilemapDto`] the Tilemap panel renders, and back-applies paint strokes
//! through [`SceneDoc::edit_paint_tiles`] (one undo step per stroke). The heavy
//! chunk map never crosses the IPC boundary — only the painted cells do.

use inf_ecs::components::CHUNK_DIM;
use uuid::Uuid;

use crate::ipc::{TilemapCellDto, TilemapDto};
use crate::scene::SceneDoc;

/// Project entity `guid`'s tilemap for the paint panel. `None` when the entity
/// has no `Tilemap`. `palette_cols`/`palette_rows` default to the map's atlas
/// grid; the Ring-2 command overrides them from the texture's sprite-sheet grid
/// when one exists (P8.2a).
pub fn build_dto(doc: &SceneDoc, guid: Uuid) -> Option<TilemapDto> {
    let tm = doc.raw_get_tilemap(guid)?;

    // Painted cells, in deterministic chunk then row-major order.
    let mut cells = Vec::new();
    for (&(cx, cy), chunk) in tm.occupied_chunks() {
        for ly in 0..CHUNK_DIM {
            for lx in 0..CHUNK_DIM {
                let idx = chunk.get(lx, ly);
                if idx != 0 {
                    cells.push(TilemapCellDto {
                        x: cx * CHUNK_DIM + lx,
                        y: cy * CHUNK_DIM + ly,
                        tile: idx,
                    });
                }
            }
        }
    }

    let atlas_cols = tm.atlas_cols.max(1);
    let atlas_rows = tm.atlas_rows.max(1);
    Some(TilemapDto {
        entity: guid.to_string(),
        texture: tm.texture.map(|u| u.to_string()),
        tile_width: tm.tile_size.x,
        tile_height: tm.tile_size.y,
        atlas_cols,
        atlas_rows,
        palette_cols: atlas_cols,
        palette_rows: atlas_rows,
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;

    #[test]
    fn build_dto_projects_painted_cells() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Tilemap, "Map", None);
        // Two cells in two different chunks.
        assert_eq!(doc.edit_paint_tiles(e, &[(0, 0, 3), (40, -1, 7)]), 2);

        let dto = build_dto(&doc, e).unwrap();
        assert_eq!(dto.atlas_cols, 4);
        assert_eq!(dto.atlas_rows, 4);
        assert_eq!(dto.palette_cols, 4);
        assert_eq!(dto.cells.len(), 2);
        assert!(dto
            .cells
            .iter()
            .any(|c| c.x == 0 && c.y == 0 && c.tile == 3));
        assert!(dto
            .cells
            .iter()
            .any(|c| c.x == 40 && c.y == -1 && c.tile == 7));
    }

    #[test]
    fn build_dto_none_for_non_tilemap() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Cube, "Cube", None);
        assert!(build_dto(&doc, e).is_none());
    }

    #[test]
    fn paint_stroke_is_one_undo_step() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Tilemap, "Map", None);
        // A stroke of several cells.
        let n = doc.edit_paint_tiles(e, &[(0, 0, 1), (1, 0, 1), (2, 0, 1)]);
        assert_eq!(n, 3);
        assert_eq!(build_dto(&doc, e).unwrap().cells.len(), 3);

        // One undo reverts the whole stroke.
        assert!(doc.undo());
        assert_eq!(build_dto(&doc, e).unwrap().cells.len(), 0);
        assert!(doc.redo());
        assert_eq!(build_dto(&doc, e).unwrap().cells.len(), 3);
    }

    #[test]
    fn painting_same_index_is_a_noop() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Tilemap, "Map", None);
        assert_eq!(doc.edit_paint_tiles(e, &[(5, 5, 2)]), 1);
        // Repainting the same value changes nothing and records no undo entry.
        assert_eq!(doc.edit_paint_tiles(e, &[(5, 5, 2)]), 0);
        assert!(doc.can_undo());
        doc.undo(); // undo the first paint
        assert_eq!(build_dto(&doc, e).unwrap().cells.len(), 0);
    }

    #[test]
    fn erasing_drops_the_cell() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Tilemap, "Map", None);
        doc.edit_paint_tiles(e, &[(0, 0, 4)]);
        assert_eq!(doc.edit_paint_tiles(e, &[(0, 0, 0)]), 1); // erase
        assert_eq!(build_dto(&doc, e).unwrap().cells.len(), 0);
        // Undo restores the erased tile.
        assert!(doc.undo());
        assert_eq!(build_dto(&doc, e).unwrap().cells[0].tile, 4);
    }
}
