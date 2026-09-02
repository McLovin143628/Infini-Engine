//! ECS facade over bevy_ecs + bevy_reflect — the only crate that names them.
//!
//! Phase 3 (ECS & scene model) foundation. The editor's scene is a bevy
//! `World` of reflected components, wrapped by [`EcsWorld`] which adds a stable
//! `Guid` index, a transform hierarchy, and dirty-gated propagation. Everything
//! the editor UI needs — snapshotting (P3.2), reflection-driven Details (P3.3),
//! undo (P3.4), and `.inf_lvl` serde (P3.5) — is built on this facade.
//!
//! Reflection note: `bevy_reflect` is used **without** its `glam` feature (that
//! pins glam 0.32, clashing with the renderer's 0.33). Editable math is stored
//! in f64 value types ([`math::Vec3d`]) that derive `Reflect` + `serde` and
//! convert to glam for computation — see [`math`].

pub mod anim;
// P29.4 the animation bridge: the Ring-0 doors the `anim.*` node kit and the
// movement step share.
pub mod anim_bridge;
pub mod attach;
// IB-2a: which static structural colliders a fixed step may hold. Sim-side by
// construction — its anchors are `StreamingSource` entities, never a camera.
pub mod band;
// P29.6: the locomotion camera's pure half. NOT sim state — it is owned by a
// host, never a component and never a resource (Ruling 4).
pub mod camera;
// P24.4: the fixed-step slot that turns a `ClothSim` into moving cloth.
pub mod cloth;
pub mod components;
// NPC1a: the sim-LOD tier system — what a crowd NPC costs this step, decided in
// ONE door both hosts call, keyed on `StreamingSource` distance and never on a
// camera. `band`'s shape with a fourth rung.
pub mod crowd;
pub mod deform;
// I6: doors — a leaf on a hinge, a lock priced as one P22 bond, and the sparse
// resource that holds what a player has done to them.
pub mod door;
// P24.4: the fixed-step slot that turns `HairGuides` into moving strands.
pub mod hair;
pub mod hierarchy;
pub mod hydro;
pub mod interact;
// I6: items, the name-keyed catalogue and the slotted inventory a character
// carries. The persistence accounting lives in the module header.
pub mod item;
pub mod math;
pub mod movement;
pub mod pose;
// SCRIPT3: the `engine.*` kit's one rule — spawn, destroy and turn — behind the
// three verbs the node kit registered at P6 and neither host implemented.
pub mod prefab;
pub mod props;
pub mod refs;
pub mod registry;
pub mod schedule;
pub mod sim;
pub mod sky;
// NPC1d: the level's own buildings, turned into people with days.
pub mod society;
pub mod traffic;
pub mod transform;
// P29.7: the raycast vehicle model — the `movement` of vehicles, with the
// fixed-step door in `inf_physics::d3::vehicle`.
pub mod vehicle;
// I6: weapons and the health they spend. Both measured in JOULES — see the
// module header for why a hit-point would put back the conversion table
// `docs/memos/p22-strength.md` exists to refuse.
pub mod weapon;
// VEN1b: the music a venue's own building implies -- an emitter entity per
// main room, reconciled against the resident volumes.
pub mod venue;
// WPN1: who saw what. A seed for the EMS arc's dispatcher, written now because
// the facts it needs live for one fixed step and are then gone.
pub mod witness;
pub mod world;

pub use attach::update_attachments;
pub use band::{
    streaming_sources, SimBand, BAND_LATTICE_M, DEFAULT_COLLIDER_FAR_M, DEFAULT_COLLIDER_NEAR_M,
};
pub use bevy_ecs::prelude::Entity;
pub use camera::{
    axis_independent_lag, blend_settings, CameraInput, CameraPose, CameraSettings, CameraTuning,
    GaitCameraSettings, LocomotionCamera, ViewMode,
};
pub use components::{
    ActorClass, AnimPlayer, AnimStateMachine, AtlasRect, AttachedTo, BillboardMode, BodyKind2D,
    BodyKind3D, Buoyancy, Camera, CharacterController2D, CharacterController3D, CharacterMovement,
    ClothSim, Collider2D, Collider3D, ColliderShape2DKind, ColliderShape3DKind, CombineRule,
    ComputedVisibility, Destructible, DoorwaySlot, Gait, GlobalTransform, Guid, HairGuides,
    IkGoalRecord, IkTarget, Joint2D, Joint3D, JointKind2D, JointKind3D, LandingKind, Light,
    LightKind, Material, MeshRef, MovementDirection, MovementMode, MovementRefusal,
    MovementRuntime, Name, PcgVolume, Primitive, ResidentSlot, RigidBody2D, RigidBody3D,
    RootMotion, RootMotionMode, RotationMode, ScatteredInstance, ScatteredSolid, SkeletalMesh,
    SkyAtmosphere, SlotRole, SmRuntimeState, SpeedCurve, Sprite, StructureGroup, Terrain,
    TileBounds, TileChunk, Tilemap, TimeOfDay, Transform, VehicleClass, Visibility, VoxelVolume,
    CHUNK_DIM, CHUNK_TILES, CRACK_OPENING_M, DEFAULT_DESTRUCTIBLE_CHUNKS,
    DEFAULT_DESTRUCTIBLE_DENSITY, DEFAULT_STRENGTH_PA,
};
// P24.2: the IK write door. `pose` and `deform` are otherwise spelled in full
// by callers (the source-text mirror gates key on that), but a *write door* a
// gameplay layer calls is a facade item like `update_attachments`.
pub use anim_bridge::{
    anim_curve, anim_param, anim_root_motion, anim_state, anim_state_is, anim_state_time,
    clear_anim_bridge, consume_anim_notify, feet_of, footstep_cues, ragdoll_rig,
    request_ragdoll_rig, set_anim_param, set_anim_trigger, set_foot_ik, set_pose_match_entry,
    take_ragdoll_rig, traversal_arc, AnimBridgeRes, AnimStateInfo, FootGoal, FootState,
    FootstepCue, RigBone, TraversalArc, FOOTSTEP_PREFIX, TRAVERSAL_ARC_SAMPLES,
};
pub use hierarchy::{ChildOf, Children};
pub use math::{Color, Vec2d, Vec3d};
pub use pose::{clear_ik_goals, ik_goals, ik_outcomes, set_ik_goals, IkGoal, IkOutcome};
// Terrain heightfield types re-exported so downstream editor crates (e.g. the
// viewport host) reach them through the ECS facade without a direct dep.
pub use inf_terrain::{HeightSource, TerrainData, TerrainTile};
pub use props::{default_field, default_list_element, ComponentProps, PropField, PropValue};
pub use refs::EntityRef;
pub use registry::{ComponentInfo, ComponentRegistry};
pub use schedule::{
    ecs_task_pool_threads, init_ecs_task_pool, ScheduleMode, SimSchedule, SimScheduleBuilder,
};
pub use sim::{sim_snapshot, AngularVelocity, EntitySimState, Lifetime, SimConfig, Velocity};
pub use sky::{
    advance_time_of_day, advance_weather, resolve_sky, sky_authority, ResolvedSky, ResolvedWeather,
    MAX_WEATHER_BLEND_S, SNOW_ACCUMULATION_MAX_M_PER_S,
};
pub use world::EcsWorld;
