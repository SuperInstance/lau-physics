//! # lau-physics
//!
//! A deterministic 2D physics engine for game worlds.
//! AABB collision detection, raycasting, and rigid body dynamics.
//! No external physics dependencies — pure Rust.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------------

/// 2D vector with `f64` components.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn dot(self, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product (scalar result: ax*by - ay*bx).
    pub fn cross(self, other: Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Vec2 {
        let len = self.length();
        if len == 0.0 {
            Vec2::zero()
        } else {
            self / len
        }
    }

    pub fn distance(self, other: Vec2) -> f64 {
        (self - other).length()
    }

    pub fn lerp(self, other: Vec2, t: f64) -> Vec2 {
        self + (other - self) * t
    }

    /// Angle in radians from the positive X axis.
    pub fn angle(self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Rotate by `angle` radians.
    pub fn rotate(self, angle: f64) -> Vec2 {
        let (sin, cos) = angle.sin_cos();
        Vec2::new(
            self.x * cos - self.y * sin,
            self.x * sin + self.y * cos,
        )
    }

    /// Perpendicular vector (rotated 90° counter-clockwise).
    pub fn perp(self) -> Vec2 {
        Vec2::new(-self.y, self.x)
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, s: f64) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
}

impl std::ops::Div<f64> for Vec2 {
    type Output = Vec2;
    fn div(self, s: f64) -> Vec2 {
        Vec2::new(self.x / s, self.y / s)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Vec2) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// Position, rotation (radians), and scale in 2D.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub position: Vec2,
    pub rotation: f64,
    pub scale: Vec2,
}

impl Transform {
    pub fn new(position: Vec2, rotation: f64, scale: Vec2) -> Self {
        Self { position, rotation, scale }
    }

    pub fn identity() -> Self {
        Self {
            position: Vec2::zero(),
            rotation: 0.0,
            scale: Vec2::new(1.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// RigidBody
// ---------------------------------------------------------------------------

/// A rigid body with linear dynamics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigidBody {
    pub transform: Transform,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub mass: f64,
    pub drag: f64,
    /// Bounciness [0, 1].
    pub restitution: f64,
    pub is_static: bool,
    pub entity_id: u64,
}

impl RigidBody {
    pub fn new(entity_id: u64) -> Self {
        Self {
            transform: Transform::identity(),
            velocity: Vec2::zero(),
            acceleration: Vec2::zero(),
            mass: 1.0,
            drag: 0.0,
            restitution: 0.5,
            is_static: false,
            entity_id,
        }
    }

    /// Apply a continuous force (stored as acceleration = force / mass).
    pub fn apply_force(&mut self, force: Vec2) {
        if self.mass > 0.0 && !self.is_static {
            self.acceleration += force / self.mass;
        }
    }

    /// Apply an instantaneous impulse to velocity.
    pub fn apply_impulse(&mut self, impulse: Vec2) {
        if self.mass > 0.0 && !self.is_static {
            self.velocity += impulse / self.mass;
        }
    }

    /// Euler integration step.
    pub fn integrate(&mut self, dt: f64) {
        if self.is_static {
            return;
        }
        // velocity += acceleration * dt
        self.velocity += self.acceleration * dt;
        // apply drag: velocity *= (1 - drag * dt)  (clamped so we don't reverse)
        let damping = (1.0 - self.drag * dt).max(0.0);
        self.velocity = self.velocity * damping;
        // position += velocity * dt
        self.transform.position += self.velocity * dt;
        // reset acceleration (forces are re-applied each frame)
        self.acceleration = Vec2::zero();
    }

    /// Compute the AABB assuming a unit half-extent centered on the body position.
    /// Callers should use their own collider shape; this is a convenience default.
    pub fn default_aabb(&self) -> AABB {
        AABB::from_center_half(self.transform.position, Vec2::new(0.5, 0.5))
    }
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub fn from_center_half(center: Vec2, half: Vec2) -> Self {
        Self {
            min: center - half,
            max: center + half,
        }
    }

    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Smallest AABB that contains both `self` and `other`.
    pub fn merge(&self, other: &AABB) -> AABB {
        AABB::new(
            Vec2::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
            ),
            Vec2::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
            ),
        )
    }

    /// Expand by `amount` in all directions.
    pub fn expand(&self, amount: f64) -> AABB {
        AABB::new(
            Vec2::new(self.min.x - amount, self.min.y - amount),
            Vec2::new(self.max.x + amount, self.max.y + amount),
        )
    }

    pub fn center(&self) -> Vec2 {
        (self.min + self.max) / 2.0
    }

    pub fn half_extents(&self) -> Vec2 {
        (self.max - self.min) / 2.0
    }

    pub fn area(&self) -> f64 {
        let d = self.max - self.min;
        d.x * d.y
    }
}

// ---------------------------------------------------------------------------
// Circle
// ---------------------------------------------------------------------------

/// Circle defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f64,
}

impl Circle {
    pub fn new(center: Vec2, radius: f64) -> Self {
        Self { center, radius }
    }

    pub fn contains_point(&self, point: Vec2) -> bool {
        self.center.distance(point) <= self.radius
    }

    pub fn intersects_circle(&self, other: &Circle) -> bool {
        self.center.distance(other.center) <= self.radius + other.radius
    }

    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        // Find the closest point on the AABB to the circle center.
        let closest_x = self.center.x.clamp(aabb.min.x, aabb.max.x);
        let closest_y = self.center.y.clamp(aabb.min.y, aabb.max.y);
        let dx = self.center.x - closest_x;
        let dy = self.center.y - closest_y;
        dx * dx + dy * dy <= self.radius * self.radius
    }
}

// ---------------------------------------------------------------------------
// Ray / RayHit
// ---------------------------------------------------------------------------

/// Infinite ray in 2D.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ray {
    pub origin: Vec2,
    pub direction: Vec2,
}

impl Ray {
    pub fn new(origin: Vec2, direction: Vec2) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Ray-AABB intersection. Returns the nearest hit (smallest positive t).
    pub fn intersect_aabb(&self, aabb: &AABB) -> Option<RayHit> {
        let inv_dir = Vec2::new(
            if self.direction.x != 0.0 { 1.0 / self.direction.x } else { f64::INFINITY },
            if self.direction.y != 0.0 { 1.0 / self.direction.y } else { f64::INFINITY },
        );

        let mut t_min = f64::NEG_INFINITY;
        let mut t_max = f64::INFINITY;
        let mut normal_min = Vec2::zero();

        // X slab
        let tx1 = (aabb.min.x - self.origin.x) * inv_dir.x;
        let tx2 = (aabb.max.x - self.origin.x) * inv_dir.x;
        let (t_near_x, t_far_x) = if tx1 < tx2 { (tx1, tx2) } else { (tx2, tx1) };
        let normal_near_x = if inv_dir.x >= 0.0 { Vec2::new(-1.0, 0.0) } else { Vec2::new(1.0, 0.0) };

        if t_near_x > t_min {
            t_min = t_near_x;
            normal_min = normal_near_x;
        }
        t_max = t_max.min(t_far_x);
        if t_min > t_max {
            return None;
        }

        // Y slab
        let ty1 = (aabb.min.y - self.origin.y) * inv_dir.y;
        let ty2 = (aabb.max.y - self.origin.y) * inv_dir.y;
        let (t_near_y, t_far_y) = if ty1 < ty2 { (ty1, ty2) } else { (ty2, ty1) };
        let normal_near_y = if inv_dir.y >= 0.0 { Vec2::new(0.0, -1.0) } else { Vec2::new(0.0, 1.0) };

        if t_near_y > t_min {
            t_min = t_near_y;
            normal_min = normal_near_y;
        }
        t_max = t_max.min(t_far_y);
        if t_min > t_max {
            return None;
        }

        let t = if t_min >= 0.0 { t_min } else if t_max >= 0.0 { t_max } else { return None };
        // Recompute normal for the chosen t
        let normal = if t == t_min { normal_min } else {
            // Hit is from the exit side; reverse the logic
            let exit_normal_x = if inv_dir.x >= 0.0 { Vec2::new(1.0, 0.0) } else { Vec2::new(-1.0, 0.0) };
            let exit_normal_y = if inv_dir.y >= 0.0 { Vec2::new(0.0, 1.0) } else { Vec2::new(0.0, -1.0) };
            if t_max == t_far_x { exit_normal_x } else { exit_normal_y }
        };

        Some(RayHit {
            point: self.origin + self.direction * t,
            normal,
            distance: t,
            entity_id: None,
        })
    }

    /// Ray-circle intersection. Returns nearest hit.
    pub fn intersect_circle(&self, circle: &Circle) -> Option<RayHit> {
        let oc = self.origin - circle.center;
        let a = self.direction.dot(self.direction); // should be 1 if normalized
        let b = 2.0 * oc.dot(self.direction);
        let c = oc.dot(oc) - circle.radius * circle.radius;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_d = discriminant.sqrt();
        let t1 = (-b - sqrt_d) / (2.0 * a);
        let t2 = (-b + sqrt_d) / (2.0 * a);
        let t = if t1 >= 0.0 { t1 } else if t2 >= 0.0 { t2 } else { return None };
        let point = self.origin + self.direction * t;
        let normal = (point - circle.center).normalize();
        Some(RayHit {
            point,
            normal,
            distance: t,
            entity_id: None,
        })
    }
}

/// Result of a ray intersection test.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RayHit {
    pub point: Vec2,
    pub normal: Vec2,
    pub distance: f64,
    pub entity_id: Option<u64>,
}

// ---------------------------------------------------------------------------
// CollisionPair
// ---------------------------------------------------------------------------

/// Describes a collision between two bodies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CollisionPair {
    pub body_a: u64,
    pub body_b: u64,
    pub normal: Vec2,
    pub penetration: f64,
    pub contact_point: Vec2,
}

// ---------------------------------------------------------------------------
// PhysicsWorld
// ---------------------------------------------------------------------------

/// The main physics simulation world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsWorld {
    pub bodies: HashMap<u64, RigidBody>,
    pub gravity: Vec2,
    pub tick: u64,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: HashMap::new(),
            gravity: Vec2::new(0.0, -9.81),
            tick: 0,
        }
    }

    pub fn add_body(&mut self, body: RigidBody) -> u64 {
        let id = body.entity_id;
        self.bodies.insert(id, body);
        id
    }

    pub fn remove_body(&mut self, id: u64) {
        self.bodies.remove(&id);
    }

    pub fn get_body(&self, id: u64) -> Option<&RigidBody> {
        self.bodies.get(&id)
    }

    pub fn get_body_mut(&mut self, id: u64) -> Option<&mut RigidBody> {
        self.bodies.get_mut(&id)
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        // 1. Apply gravity & integrate
        let gravity = self.gravity;
        for body in self.bodies.values_mut() {
            if !body.is_static {
                body.apply_force(gravity * body.mass);
            }
            body.integrate(dt);
        }

        // 2. Detect AABB collisions
        let collision_pairs = self.detect_collisions();

        // 3. Resolve collisions
        for pair in &collision_pairs {
            self.resolve_collision(pair);
        }

        self.tick += 1;
    }

    /// Broad-phase AABB collision detection for all body pairs.
    fn detect_collisions(&self) -> Vec<CollisionPair> {
        let mut pairs = Vec::new();
        let ids: Vec<u64> = self.bodies.keys().copied().collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let id_a = ids[i];
                let id_b = ids[j];
                let body_a = &self.bodies[&id_a];
                let body_b = &self.bodies[&id_b];
                let aabb_a = body_a.default_aabb();
                let aabb_b = body_b.default_aabb();
                if aabb_a.intersects(&aabb_b) {
                    let overlap_x = aabb_a.max.x.min(aabb_b.max.x) - aabb_a.min.x.max(aabb_b.min.x);
                    let overlap_y = aabb_a.max.y.min(aabb_b.max.y) - aabb_a.min.y.max(aabb_b.min.y);
                    let (penetration, normal) = if overlap_x < overlap_y {
                        // Normal points from A towards B
                        let sign = if body_a.transform.position.x < body_b.transform.position.x { 1.0 } else { -1.0 };
                        (overlap_x, Vec2::new(sign, 0.0))
                    } else {
                        let sign = if body_a.transform.position.y < body_b.transform.position.y { 1.0 } else { -1.0 };
                        (overlap_y, Vec2::new(0.0, sign))
                    };
                    let contact = aabb_a.center().lerp(aabb_b.center(), 0.5);
                    pairs.push(CollisionPair {
                        body_a: id_a,
                        body_b: id_b,
                        normal,
                        penetration,
                        contact_point: contact,
                    });
                }
            }
        }
        pairs
    }

    /// Resolve a collision by separating bodies and applying impulses.
    fn resolve_collision(&mut self, pair: &CollisionPair) {
        let restitution = {
            let body_a = &self.bodies[&pair.body_a];
            let body_b = &self.bodies[&pair.body_b];
            body_a.restitution.min(body_b.restitution)
        };

        // Separate bodies along the collision normal
        let (static_a, static_b, inv_mass_a, inv_mass_b) = {
            let body_a = &self.bodies[&pair.body_a];
            let body_b = &self.bodies[&pair.body_b];
            let inv_a = if body_a.is_static { 0.0 } else { 1.0 / body_a.mass };
            let inv_b = if body_b.is_static { 0.0 } else { 1.0 / body_b.mass };
            (body_a.is_static, body_b.is_static, inv_a, inv_b)
        };

        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass == 0.0 {
            return;
        }

        // Positional correction (separation)
        let correction = pair.normal * (pair.penetration / total_inv_mass);
        if !static_a {
            self.bodies.get_mut(&pair.body_a).unwrap().transform.position -= correction * inv_mass_a;
        }
        if !static_b {
            self.bodies.get_mut(&pair.body_b).unwrap().transform.position += correction * inv_mass_b;
        }

        // Impulse resolution
        let rel_vel = {
            let body_a = &self.bodies[&pair.body_a];
            let body_b = &self.bodies[&pair.body_b];
            body_b.velocity - body_a.velocity
        };

        let vel_along_normal = rel_vel.dot(pair.normal);
        // Only resolve if bodies are moving towards each other
        if vel_along_normal > 0.0 {
            return;
        }

        let j = -(1.0 + restitution) * vel_along_normal / total_inv_mass;
        let impulse = pair.normal * j;

        if !static_a {
            let body_a = self.bodies.get_mut(&pair.body_a).unwrap();
            body_a.velocity -= impulse * inv_mass_a;
        }
        if !static_b {
            let body_b = self.bodies.get_mut(&pair.body_b).unwrap();
            body_b.velocity += impulse * inv_mass_b;
        }
    }

    /// Cast a ray against all bodies, returning sorted hits.
    pub fn cast_ray(&self, ray: &Ray) -> Vec<RayHit> {
        let mut hits = Vec::new();
        for (&id, body) in &self.bodies {
            let aabb = body.default_aabb();
            if let Some(mut hit) = ray.intersect_aabb(&aabb) {
                hit.entity_id = Some(id);
                hits.push(hit);
            }
        }
        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Return IDs of all bodies whose default AABB overlaps the given AABB.
    pub fn query_aabb(&self, aabb: &AABB) -> Vec<u64> {
        self.bodies
            .iter()
            .filter(|(_, body)| body.default_aabb().intersects(aabb))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Return IDs of all bodies whose default AABB overlaps the given circle.
    pub fn query_circle(&self, circle: &Circle) -> Vec<u64> {
        self.bodies
            .iter()
            .filter(|(_, body)| circle.intersects_aabb(&body.default_aabb()))
            .map(|(&id, _)| id)
            .collect()
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Vec2 ----

    #[test]
    fn vec2_new_and_zero() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
        assert_eq!(Vec2::zero(), Vec2::new(0.0, 0.0));
    }

    #[test]
    fn vec2_dot_and_cross() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), 1.0);
    }

    #[test]
    fn vec2_length_and_normalize() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-10);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-10);
        assert_eq!(Vec2::zero().normalize(), Vec2::zero());
    }

    #[test]
    fn vec2_distance_and_lerp() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 0.0);
        assert!((a.distance(b) - 10.0).abs() < 1e-10);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn vec2_rotate_and_perp() {
        let v = Vec2::new(1.0, 0.0);
        let rotated = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!((rotated.x - 0.0).abs() < 1e-10);
        assert!((rotated.y - 1.0).abs() < 1e-10);
        let p = v.perp();
        assert_eq!(p, Vec2::new(0.0, 1.0));
    }

    #[test]
    fn vec2_angle() {
        let v = Vec2::new(1.0, 0.0);
        assert!((v.angle() - 0.0).abs() < 1e-10);
        let up = Vec2::new(0.0, 1.0);
        assert!((up.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn vec2_arithmetic() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a + b, Vec2::new(4.0, 6.0));
        assert_eq!(b - a, Vec2::new(2.0, 2.0));
        assert_eq!(a * 2.0, Vec2::new(2.0, 4.0));
        assert_eq!(a / 2.0, Vec2::new(0.5, 1.0));
        assert_eq!(-a, Vec2::new(-1.0, -2.0));
    }

    // ---- AABB ----

    #[test]
    fn aabb_from_center_half() {
        let aabb = AABB::from_center_half(Vec2::new(5.0, 5.0), Vec2::new(1.0, 2.0));
        assert_eq!(aabb.min, Vec2::new(4.0, 3.0));
        assert_eq!(aabb.max, Vec2::new(6.0, 7.0));
    }

    #[test]
    fn aabb_contains_point() {
        let aabb = AABB::from_center_half(Vec2::new(5.0, 5.0), Vec2::new(2.0, 2.0));
        assert!(aabb.contains_point(Vec2::new(5.0, 5.0)));
        assert!(aabb.contains_point(Vec2::new(3.0, 3.0)));
        assert!(!aabb.contains_point(Vec2::new(0.0, 0.0)));
    }

    #[test]
    fn aabb_intersects() {
        let a = AABB::from_center_half(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let b = AABB::from_center_half(Vec2::new(1.5, 0.0), Vec2::new(1.0, 1.0));
        assert!(a.intersects(&b));
        let c = AABB::from_center_half(Vec2::new(5.0, 5.0), Vec2::new(0.5, 0.5));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn aabb_merge_expand_area() {
        let a = AABB::from_center_half(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let b = AABB::from_center_half(Vec2::new(4.0, 4.0), Vec2::new(1.0, 1.0));
        let merged = a.merge(&b);
        assert_eq!(merged.min, Vec2::new(-1.0, -1.0));
        assert_eq!(merged.max, Vec2::new(5.0, 5.0));
        let expanded = a.expand(1.0);
        assert_eq!(expanded.min, Vec2::new(-2.0, -2.0));
        assert_eq!(expanded.max, Vec2::new(2.0, 2.0));
        assert!((a.area() - 4.0).abs() < 1e-10);
        assert_eq!(a.center(), Vec2::new(0.0, 0.0));
        assert_eq!(a.half_extents(), Vec2::new(1.0, 1.0));
    }

    // ---- Circle ----

    #[test]
    fn circle_contains_and_intersects() {
        let c = Circle::new(Vec2::new(0.0, 0.0), 5.0);
        assert!(c.contains_point(Vec2::new(3.0, 4.0)));
        assert!(!c.contains_point(Vec2::new(5.0, 5.0)));
        let d = Circle::new(Vec2::new(8.0, 0.0), 4.0);
        assert!(c.intersects_circle(&d));
        let e = Circle::new(Vec2::new(20.0, 0.0), 1.0);
        assert!(!c.intersects_circle(&e));
    }

    #[test]
    fn circle_intersects_aabb() {
        let circle = Circle::new(Vec2::new(5.0, 0.0), 3.0);
        let aabb = AABB::from_center_half(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        assert!(circle.intersects_aabb(&aabb));
        let far = AABB::from_center_half(Vec2::new(20.0, 20.0), Vec2::new(1.0, 1.0));
        assert!(!circle.intersects_aabb(&far));
    }

    // ---- Ray ----

    #[test]
    fn ray_intersect_aabb_hit() {
        let ray = Ray::new(Vec2::new(-5.0, 0.0), Vec2::new(1.0, 0.0));
        let aabb = AABB::from_center_half(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let hit = ray.intersect_aabb(&aabb).unwrap();
        assert!((hit.distance - 4.0).abs() < 1e-10);
        assert!((hit.point.x - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn ray_intersect_aabb_miss() {
        let ray = Ray::new(Vec2::new(-5.0, 10.0), Vec2::new(1.0, 0.0));
        let aabb = AABB::from_center_half(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        assert!(ray.intersect_aabb(&aabb).is_none());
    }

    #[test]
    fn ray_intersect_circle_hit() {
        let ray = Ray::new(Vec2::new(-5.0, 0.0), Vec2::new(1.0, 0.0));
        let circle = Circle::new(Vec2::new(0.0, 0.0), 1.0);
        let hit = ray.intersect_circle(&circle).unwrap();
        assert!((hit.distance - 4.0).abs() < 1e-10);
    }

    #[test]
    fn ray_intersect_circle_miss() {
        let ray = Ray::new(Vec2::new(-5.0, 10.0), Vec2::new(1.0, 0.0));
        let circle = Circle::new(Vec2::new(0.0, 0.0), 1.0);
        assert!(ray.intersect_circle(&circle).is_none());
    }

    #[test]
    fn ray_inside_aabb() {
        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let aabb = AABB::from_center_half(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let hit = ray.intersect_aabb(&aabb).unwrap();
        assert!(hit.distance >= 0.0);
        assert!((hit.distance - 2.0).abs() < 1e-10);
    }

    // ---- RigidBody ----

    #[test]
    fn rigid_body_integrate() {
        let mut body = RigidBody::new(1);
        body.velocity = Vec2::new(1.0, 0.0);
        body.acceleration = Vec2::new(0.0, 1.0);
        body.integrate(1.0);
        assert!((body.velocity.y - 1.0).abs() < 1e-10);
        assert!((body.transform.position.x - 1.0).abs() < 1e-10);
        assert!((body.transform.position.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rigid_body_static_no_move() {
        let mut body = RigidBody::new(2);
        body.is_static = true;
        body.velocity = Vec2::new(10.0, 10.0);
        body.integrate(1.0);
        assert_eq!(body.transform.position, Vec2::zero());
    }

    #[test]
    fn rigid_body_drag() {
        let mut body = RigidBody::new(3);
        body.velocity = Vec2::new(10.0, 0.0);
        body.drag = 0.5;
        body.integrate(1.0);
        assert!(body.velocity.x < 10.0);
        assert!(body.velocity.x > 0.0);
    }

    #[test]
    fn apply_force_and_impulse() {
        let mut body = RigidBody::new(4);
        body.mass = 2.0;
        body.apply_force(Vec2::new(4.0, 0.0));
        assert!((body.acceleration.x - 2.0).abs() < 1e-10);
        body.apply_impulse(Vec2::new(6.0, 0.0));
        assert!((body.velocity.x - 3.0).abs() < 1e-10);
    }

    // ---- PhysicsWorld ----

    #[test]
    fn world_add_remove_get() {
        let mut world = PhysicsWorld::new();
        let body = RigidBody::new(42);
        world.add_body(body);
        assert!(world.get_body(42).is_some());
        world.remove_body(42);
        assert!(world.get_body(42).is_none());
    }

    #[test]
    fn world_gravity_simulation() {
        let mut world = PhysicsWorld::new();
        let mut body = RigidBody::new(1);
        body.mass = 1.0;
        world.add_body(body);
        world.step(1.0);
        let b = world.get_body(1).unwrap();
        // After 1s of gravity (0, -9.81), position y should be negative
        assert!(b.transform.position.y < 0.0);
    }

    #[test]
    fn world_collision_resolution() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec2::zero();

        let mut a = RigidBody::new(1);
        a.transform.position = Vec2::new(-0.4, 0.0);
        a.velocity = Vec2::new(1.0, 0.0);
        a.mass = 1.0;
        a.restitution = 1.0;
        world.add_body(a);

        let mut b = RigidBody::new(2);
        b.transform.position = Vec2::new(0.4, 0.0);
        b.velocity = Vec2::new(-1.0, 0.0);
        b.mass = 1.0;
        b.restitution = 1.0;
        world.add_body(b);

        world.step(0.01);
        let ba = world.get_body(1).unwrap();
        let bb = world.get_body(2).unwrap();
        // After collision with restitution=1, bodies should bounce apart
        assert!(ba.velocity.x < 0.0, "body A should bounce left, got vel={}", ba.velocity.x);
        assert!(bb.velocity.x > 0.0, "body B should bounce right, got vel={}", bb.velocity.x);
    }

    #[test]
    fn conservation_of_momentum() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec2::zero();

        let mut a = RigidBody::new(1);
        a.transform.position = Vec2::new(-0.4, 0.0);
        a.velocity = Vec2::new(3.0, 0.0);
        a.mass = 2.0;
        a.restitution = 1.0;

        let mut b = RigidBody::new(2);
        b.transform.position = Vec2::new(0.4, 0.0);
        b.velocity = Vec2::new(-1.0, 0.0);
        b.mass = 1.0;
        b.restitution = 1.0;

        // Initial momentum: 2*3 + 1*(-1) = 5
        let p_before = a.mass * a.velocity.x + b.mass * b.velocity.x;

        world.add_body(a);
        world.add_body(b);

        world.step(0.1);

        let ba = world.get_body(1).unwrap();
        let bb = world.get_body(2).unwrap();
        let p_after = ba.mass * ba.velocity.x + bb.mass * bb.velocity.x;
        assert!(
            (p_before - p_after).abs() < 0.5,
            "momentum should be roughly conserved: before={}, after={}",
            p_before, p_after
        );
    }

    #[test]
    fn world_cast_ray() {
        let mut world = PhysicsWorld::new();
        let mut body = RigidBody::new(10);
        body.transform.position = Vec2::new(5.0, 0.0);
        world.add_body(body);

        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let hits = world.cast_ray(&ray);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entity_id, Some(10));
        assert!(hits[0].distance > 0.0);
    }

    #[test]
    fn world_query_aabb() {
        let mut world = PhysicsWorld::new();
        let mut body = RigidBody::new(1);
        body.transform.position = Vec2::new(5.0, 5.0);
        world.add_body(body);

        let query = AABB::from_center_half(Vec2::new(5.0, 5.0), Vec2::new(1.0, 1.0));
        let ids = world.query_aabb(&query);
        assert!(ids.contains(&1));

        let far = AABB::from_center_half(Vec2::new(100.0, 100.0), Vec2::new(1.0, 1.0));
        assert!(world.query_aabb(&far).is_empty());
    }

    #[test]
    fn world_query_circle() {
        let mut world = PhysicsWorld::new();
        let mut body = RigidBody::new(1);
        body.transform.position = Vec2::new(5.0, 5.0);
        world.add_body(body);

        let circle = Circle::new(Vec2::new(5.0, 5.0), 1.0);
        let ids = world.query_circle(&circle);
        assert!(ids.contains(&1));

        let far = Circle::new(Vec2::new(100.0, 100.0), 1.0);
        assert!(world.query_circle(&far).is_empty());
    }

    #[test]
    fn world_tick_increments() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec2::zero();
        world.add_body(RigidBody::new(1));
        assert_eq!(world.tick, 0);
        world.step(0.016);
        assert_eq!(world.tick, 1);
        world.step(0.016);
        assert_eq!(world.tick, 2);
    }

    #[test]
    fn serde_roundtrip() {
        let mut world = PhysicsWorld::new();
        let mut body = RigidBody::new(1);
        body.transform.position = Vec2::new(1.0, 2.0);
        body.velocity = Vec2::new(3.0, 4.0);
        world.add_body(body);

        let json = serde_json::to_string(&world).unwrap();
        let world2: PhysicsWorld = serde_json::from_str(&json).unwrap();
        assert_eq!(world.bodies.len(), world2.bodies.len());
        let b1 = world.get_body(1).unwrap();
        let b2 = world2.get_body(1).unwrap();
        assert_eq!(b1.transform.position, b2.transform.position);
        assert_eq!(b1.velocity, b2.velocity);
    }

    #[test]
    fn static_body_blocks() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec2::zero();

        let mut floor = RigidBody::new(99);
        floor.transform.position = Vec2::new(0.0, -2.0);
        floor.is_static = true;
        floor.restitution = 1.0;
        world.add_body(floor);

        let mut ball = RigidBody::new(1);
        ball.transform.position = Vec2::new(0.0, -0.9);
        ball.velocity = Vec2::new(0.0, -5.0);
        ball.mass = 1.0;
        ball.restitution = 1.0;
        world.add_body(ball);

        world.step(0.05);
        let b = world.get_body(1).unwrap();
        // Ball should bounce off the floor (velocity should be positive/upward after collision)
        assert!(b.velocity.y > 0.0, "ball should bounce upward, got vel.y={}", b.velocity.y);
    }

    #[test]
    fn multiple_raycast_hits_sorted() {
        let mut world = PhysicsWorld::new();
        let mut a = RigidBody::new(1);
        a.transform.position = Vec2::new(3.0, 0.0);
        world.add_body(a);

        let mut b = RigidBody::new(2);
        b.transform.position = Vec2::new(7.0, 0.0);
        world.add_body(b);

        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let hits = world.cast_ray(&ray);
        assert_eq!(hits.len(), 2);
        assert!(hits[0].distance < hits[1].distance);
        assert_eq!(hits[0].entity_id, Some(1));
        assert_eq!(hits[1].entity_id, Some(2));
    }

    #[test]
    fn transform_identity() {
        let t = Transform::identity();
        assert_eq!(t.position, Vec2::zero());
        assert_eq!(t.rotation, 0.0);
        assert_eq!(t.scale, Vec2::new(1.0, 1.0));
    }

    #[test]
    fn collision_pair_fields() {
        let pair = CollisionPair {
            body_a: 1,
            body_b: 2,
            normal: Vec2::new(1.0, 0.0),
            penetration: 0.5,
            contact_point: Vec2::new(0.0, 0.0),
        };
        assert_eq!(pair.body_a, 1);
        assert_eq!(pair.body_b, 2);
        assert_eq!(pair.normal, Vec2::new(1.0, 0.0));
    }
}
