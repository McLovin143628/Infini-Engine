//! **THE ART TABLE** (wave VEH3f.2a) -- every imported vehicle body a roster
//! row can name, as DATA: `vehicle_art.toml`, parsed once.
//!
//! # Why a table and not an enum
//!
//! Wave VEH3f's `ArtKey` was a closed seven-variant enum matched against the
//! construction pack's static-mesh names, so a car pack could not be named by
//! any row without a code change in three crates (`roster.rs`, the importer's
//! match, the fallback generator). The key is now an index into this table: a
//! row's `art = "dd_sedan"` resolves by name, the seven machines resolve
//! exactly as before (same names, so the same GUIDs), and a new body is a table
//! row plus the measured numbers the importer printed for it.
//!
//! # What a row carries -- NUMBERS, never content
//!
//! Everything below is a measurement the VEH3f.2a importer took off the pack
//! (a bone's rest position, a part's bounds, a Blueprint default) and printed
//! into `<key>.vehicle.toml` beside the local art. No vertex, texel or name of a
//! pack asset beyond its source mesh's stem is here; the art itself lives in a
//! LOCAL project at the GUIDs [`art_body_guid`] / [`art_wheel_guid`] /
//! [`art_part_guid`] derive, and the engine repository commits a fallback of
//! its own geometry at the same GUIDs (`samples/vehicle-art/`).
//!
//! # The art's own parts
//!
//! A car pack whose doors are skinned to hinge bones crosses as separate door
//! meshes, and a steering wheel as a separate mesh: each becomes a [`BodyPart`]
//! of the row (named by the one naming rule, [`BodyPartKind::of`], so the world
//! recognises what the table declares). A door's box is chosen so its FRONT
//! EDGE and its lateral centre sit on the pack's hinge bone -- which is where
//! [`crate::vehicle::Hinge::of`] puts every door's axis -- so the VEH3c revolute
//! swings the art about the art's own pivot with no second hinge rule. A body
//! whose table row names no parts (the seven machines) keeps its family's seats,
//! exactly as VEH3f drew it.

use std::sync::OnceLock;

use crate::math::Vec3d;
use crate::vehicle::{BodyPart, BodyPartKind};

/// The committed art table.
pub const VEHICLE_ART_TOML: &str = include_str!("vehicle_art.toml");

/// **An imported vehicle body**, by its index in the art table.
///
/// `Copy` and ordered, so a [`crate::vehicle::VehicleDef`] can carry one; the
/// stable identity is its [`name`](Self::name), which every GUID is derived
/// from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtKey(u16);

/// How a pack's vehicles face in Unreal's own frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Facing {
    /// UE `+X` -- every car pack the bridge has met.
    PlusX,
    /// UE `+Y` -- the construction pack (wave VEH3f).
    PlusY,
}

/// One row of the art table.
#[derive(Clone, Debug)]
pub struct ArtBody {
    /// The stable key a row's `art = "..."` spells.
    pub key: &'static str,
    /// The Unreal mesh the body is split from (its stem).
    pub source: &'static str,
    /// The pack it came from.
    pub pack: &'static str,
    /// How the pack faces.
    pub facing: Facing,
    /// Tracks rather than wheels (the tracks stay in the body).
    pub tracked: bool,
    /// The art's own collider half-extents, metres, when the table records
    /// them -- a row that names this art must author exactly these.
    pub half_extents: Option<Vec3d>,
    /// The art-derived parts (doors, panes, the steering wheel, the seats), in
    /// fractions of [`half_extents`](Self::half_extents); empty for a body that
    /// keeps its family's seats.
    pub parts: &'static [BodyPart],
    /// The pack has no door meshes: the doors in [`parts`](Self::parts) are
    /// INVISIBLE proxies over fused art (stated, and priced in the wave report).
    pub door_proxy: bool,
    /// The in-car camera the pack's Blueprint suggests, chassis frame, metres.
    pub camera: Option<Vec3d>,
    /// The drawn steering wheel's column rake, degrees, in
    /// `hub_rim_euler`'s convention -- the pack's own, measured.
    pub hub_rake_deg: Option<f64>,
    /// How many LOD rungs the pack ships for this body (the engine draws a
    /// meshlet DAG, so these are recorded, not stored).
    pub pack_lods: u32,
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn vec3(v: Option<&toml::Value>) -> Option<Vec3d> {
    let a = v?.as_array()?;
    if a.len() != 3 {
        return None;
    }
    let f = |x: &toml::Value| x.as_float().or_else(|| x.as_integer().map(|i| i as f64));
    Some(Vec3d::new(f(&a[0])?, f(&a[1])?, f(&a[2])?))
}

/// Parse an art table -- the committed one, or a test's.
pub fn parse_art_table(text: &str) -> Result<Vec<ArtBody>, String> {
    let doc: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
    let rows = doc
        .get("art")
        .and_then(|a| a.as_array())
        .ok_or_else(|| "the art table is `[[art]]` rows".to_string())?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let s = |k: &str| row.get(k).and_then(|v| v.as_str());
        let key = s("key").ok_or("an art row has no `key`")?;
        if out.iter().any(|b: &ArtBody| b.key == key) {
            return Err(format!("art key `{key}` twice"));
        }
        let facing = match s("facing").unwrap_or("+X") {
            "+X" => Facing::PlusX,
            "+Y" => Facing::PlusY,
            f => return Err(format!("art `{key}`: unknown facing `{f}`")),
        };
        let half_extents = vec3(row.get("half_extents_m"));
        let mut parts = Vec::new();
        for p in row
            .get("parts")
            .and_then(|p| p.as_array())
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            let name = p
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("art `{key}`: a part has no name"))?;
            let centre = vec3(p.get("centre"))
                .ok_or_else(|| format!("art `{key}` part `{name}`: no centre"))?;
            let half =
                vec3(p.get("half")).ok_or_else(|| format!("art `{key}` part `{name}`: no half"))?;
            parts.push(BodyPart {
                name: leak(name),
                centre,
                half,
                primitive: crate::components::Primitive::Cube,
                kind: BodyPartKind::of(name, centre, half),
            });
        }
        if !parts.is_empty() && half_extents.is_none() {
            return Err(format!(
                "art `{key}` has parts but no `half_extents_m` -- its fractions mean nothing"
            ));
        }
        out.push(ArtBody {
            key: leak(key),
            source: leak(s("source").unwrap_or("")),
            pack: leak(s("pack").unwrap_or("")),
            facing,
            tracked: row
                .get("tracked")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            half_extents,
            parts: Box::leak(parts.into_boxed_slice()),
            door_proxy: row
                .get("door_proxy")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            camera: vec3(row.get("camera_m")),
            hub_rake_deg: row
                .get("hub_rake_deg")
                .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64))),
            pack_lods: row
                .get("pack_lods")
                .and_then(|v| v.as_integer())
                .unwrap_or(1)
                .max(1) as u32,
        });
    }
    if out.len() > u16::MAX as usize {
        return Err("the art table is too long".to_string());
    }
    Ok(out)
}

/// The committed table, parsed once.
pub fn art_table() -> &'static [ArtBody] {
    static TABLE: OnceLock<Vec<ArtBody>> = OnceLock::new();
    TABLE.get_or_init(|| parse_art_table(VEHICLE_ART_TOML).expect("vehicle_art.toml parses"))
}

/// The seven construction machines wave VEH3f imported, by key.
pub const MACHINES: [&str; 7] = [
    "excavator",
    "dozer",
    "dump_truck",
    "mixer",
    "forklift",
    "crane_truck",
    "freight_tractor",
];

impl ArtKey {
    /// Every body in the table, in table order.
    pub fn all() -> Vec<ArtKey> {
        (0..art_table().len() as u16).map(ArtKey).collect()
    }

    /// The seven construction machines, in VEH3f's order.
    pub fn machines() -> Vec<ArtKey> {
        MACHINES
            .iter()
            .filter_map(|m| ArtKey::from_name(m))
            .collect()
    }

    /// This body's table row.
    pub fn body(self) -> &'static ArtBody {
        &art_table()[self.0 as usize]
    }

    /// The stable key a row's `art = "..."` spells and the files are named by.
    pub fn name(self) -> &'static str {
        self.body().key
    }

    /// The key a name means, or `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        art_table()
            .iter()
            .position(|b| b.key == name)
            .map(|i| ArtKey(i as u16))
    }

    /// The Unreal mesh this key is split from.
    pub fn source_mesh(self) -> &'static str {
        self.body().source
    }

    /// Whether the machine runs on tracks -- its "wheels" are track frames,
    /// which stay in the body mesh and do not spin.
    pub fn tracked(self) -> bool {
        self.body().tracked
    }

    /// The art-derived parts, or empty for a body that keeps its family's seats.
    pub fn parts(self) -> &'static [BodyPart] {
        self.body().parts
    }
}

/// The salt of the art guids -- `"VEH3FARTBODYMESH"` in ASCII.
const ART_SALT: u128 = 0x5645_4833_4641_5254_424f_4459_4d45_5348;

/// **An art GUID by NAME** -- the importer's door, which knows the key it is
/// writing before any table row exists for it.
pub fn art_guid_named(key: &str, part: &str) -> uuid::Uuid {
    let mut x = ART_SALT;
    for b in key.as_bytes().iter().chain(b"/").chain(part.as_bytes()) {
        x = x.rotate_left(7) ^ (*b as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    x = x.rotate_left(33) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    uuid::Builder::from_random_bytes(x.to_be_bytes()).into_uuid()
}

/// **The body mesh a machine's rows draw**, a pure function of its key.
pub fn art_body_guid(key: ArtKey) -> uuid::Uuid {
    art_guid_named(key.name(), "body")
}

/// **The mesh rig wheel `i` of a machine draws** (front left, front right,
/// rear left, rear right -- `VehicleDef::wheel_mounts`' order).
pub fn art_wheel_guid(key: ArtKey, i: usize) -> uuid::Uuid {
    art_guid_named(key.name(), &format!("wheel{i}"))
}

/// **The mesh one art part draws** (wave VEH3f.2a) -- a door, a pane, the
/// steering wheel, a seat -- by the part's name.
pub fn art_part_guid(key: ArtKey, part: &str) -> uuid::Uuid {
    art_guid_named(key.name(), part)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seven machines resolve, by the names (and so the GUIDs) wave VEH3f
    /// committed, and every table row's parts are recognised by the naming
    /// rule as the kind the table declares.
    #[test]
    fn the_art_table_parses_and_the_machines_keep_their_names() {
        let t = art_table();
        assert!(t.len() >= 7);
        for m in MACHINES {
            let k = ArtKey::from_name(m).expect(m);
            assert_eq!(k.name(), m);
            assert!(k.parts().is_empty(), "{m} keeps its family's seats");
        }
        assert_eq!(
            ArtKey::from_name("excavator").map(|k| k.tracked()),
            Some(true)
        );
        for key in ArtKey::all() {
            for p in key.parts() {
                assert_eq!(
                    BodyPartKind::of(p.name, p.centre, p.half),
                    p.kind,
                    "{} {}",
                    key.name(),
                    p.name
                );
            }
        }
        assert!(parse_art_table("[[art]]\nkey = \"a\"\n[[art]]\nkey = \"a\"\n").is_err());
    }
}
