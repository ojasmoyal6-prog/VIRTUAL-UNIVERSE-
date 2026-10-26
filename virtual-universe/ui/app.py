"""Main application: layout, panels, interaction loop."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pygame

from experiments.scenarios import (
    SCENARIO_LOADERS,
    load_random_universe,
    load_scenario,
    load_solar_system,
)
from simulation.bodies import Body, BodyType
from simulation.collisions import CollisionMode
from simulation.constants import AU, DAY, M_EARTH, M_SUN, R_EARTH, R_SUN
from simulation.integrator import IntegratorKind
from simulation.orbital import analyze_orbit, format_period
from simulation.universe import Universe
from ui.physics_docs import EQUATIONS_SHORT, PHYSICS_ABOUT
from ui.widgets import UIContext
from utils.export import export_csv, export_experiment_report
from utils.formatting import fmt_distance, fmt_mass, fmt_speed, fmt_time_days, sci
from utils.learn import learn_messages
from visualization.camera import Camera
from visualization.renderer import UniverseRenderer
from visualization.theme import THEME

SPEEDS = [0.1, 1, 10, 100, 1000, 10000, 1_000_000]
PANEL_W = 340
TOP_H = 44
BOTTOM_H = 56


class App:
    def __init__(self, width: int = 1400, height: int = 900) -> None:
        pygame.init()
        pygame.display.set_caption("Virtual Universe Simulator")
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo", 16)
        self.font_small = pygame.font.SysFont("menlo", 13)
        self.font_title = pygame.font.SysFont("menlo", 18, bold=True)
        self.ui = UIContext(self.font, self.font_small)

        self.universe = Universe(dt=3600.0, integrator=IntegratorKind.VELOCITY_VERLET)
        self.camera = Camera(mode="2d", zoom=0.8)
        self.renderer = UniverseRenderer()
        self.right_tab = "object"  # object | physics | events | graphs | builder | about
        self.scenario_names = list(SCENARIO_LOADERS.keys())
        self.scenario_index = 0
        self.speed_index = 1
        self.edit_fields: Dict[str, str] = {}
        self.graph_surface: Optional[pygame.Surface] = None
        self._drag_pan = False
        self._drag_rot = False
        self._last_mouse = (0, 0)
        self.builder_type_index = 0
        self.body_types = list(BodyType)
        self.collision_modes = list(CollisionMode)
        self.collision_index = 0
        self.random_cfg = {
            "stars": "2",
            "planets": "8",
            "moons": "4",
        }
        self.export_dir = Path.cwd() / "exports"
        self.export_dir.mkdir(exist_ok=True)
        self.status_message = "Load a scenario or build your own universe."
        self.about_scroll = 0

        load_solar_system(self.universe)
        self.camera.fit_bodies(self.universe.bodies, self._view_size())
        self.universe.play()

    # ------------------------------------------------------------------
    # Layout geometry
    # ------------------------------------------------------------------
    def _view_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(0, TOP_H, max(100, w - PANEL_W), max(100, h - TOP_H - BOTTOM_H))

    def _view_size(self):
        r = self._view_rect()
        return (r.width, r.height)

    def _panel_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(w - PANEL_W, TOP_H, PANEL_W, h - TOP_H - BOTTOM_H)

    def _bottom_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(0, h - BOTTOM_H, w, BOTTOM_H)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        while True:
            dt = self.clock.tick(60) / 1000.0
            events = pygame.event.get()
            if not self._handle_events(events):
                break
            self.ui.begin_frame(events)

            # Camera follow
            if self.universe.focus_id:
                body = self.universe.get_body(self.universe.focus_id)
                self.camera.update_follow(body)

            self.universe.advance(dt)
            self._draw()
            pygame.display.flip()
        pygame.quit()

    def _handle_events(self, events: List[pygame.event.Event]) -> bool:
        view = self._view_rect()
        for e in events:
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                if e.key == pygame.K_SPACE:
                    self.universe.toggle()
                if e.key == pygame.K_r:
                    self.universe.reset_to_snapshot()
                if e.key == pygame.K_f:
                    self.camera.fit_bodies(self.universe.bodies, self._view_size())
                if e.key == pygame.K_1:
                    self.camera.mode = "2d"
                if e.key == pygame.K_2:
                    self.camera.mode = "3d"
                if e.key == pygame.K_TAB:
                    tabs = ["object", "builder", "physics", "events", "graphs", "about"]
                    i = tabs.index(self.right_tab) if self.right_tab in tabs else 0
                    self.right_tab = tabs[(i + 1) % len(tabs)]
            if e.type == pygame.MOUSEBUTTONDOWN:
                if view.collidepoint(e.pos):
                    local = (e.pos[0] - view.x, e.pos[1] - view.y)
                    if e.button == 1:
                        picked = self.renderer.pick_body(
                            self.universe, self.camera, self._view_size(), local[0], local[1]
                        )
                        if picked:
                            self.universe.selected_id = picked
                            self._sync_edit_fields()
                        else:
                            self._drag_pan = True
                            self._last_mouse = e.pos
                    elif e.button == 3:
                        self._drag_rot = True
                        self._last_mouse = e.pos
                    elif e.button == 4:
                        self.camera.zoom_at(1.15, local[0], local[1], self._view_size())
                    elif e.button == 5:
                        self.camera.zoom_at(1 / 1.15, local[0], local[1], self._view_size())
            if e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    self._drag_pan = False
                if e.button == 3:
                    self._drag_rot = False
            if e.type == pygame.MOUSEMOTION:
                if self._drag_pan:
                    dx = e.pos[0] - self._last_mouse[0]
                    dy = e.pos[1] - self._last_mouse[1]
                    self.camera.pan_pixels(dx, dy)
                    self._last_mouse = e.pos
                if self._drag_rot and self.camera.mode == "3d":
                    dx = e.pos[0] - self._last_mouse[0]
                    dy = e.pos[1] - self._last_mouse[1]
                    self.camera.yaw += dx * 0.005
                    self.camera.pitch = float(
                        np.clip(self.camera.pitch + dy * 0.005, -1.4, 1.4)
                    )
                    self._last_mouse = e.pos
        return True

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw(self) -> None:
        self.screen.fill(THEME.bg)
        self._draw_top_bar()
        self._draw_universe_view()
        self._draw_side_panel()
        self._draw_bottom_bar()

    def _draw_top_bar(self) -> None:
        w, _ = self.screen.get_size()
        bar = pygame.Rect(0, 0, w, TOP_H)
        pygame.draw.rect(self.screen, THEME.panel, bar)
        pygame.draw.line(self.screen, THEME.panel_border, (0, TOP_H - 1), (w, TOP_H - 1))
        title = self.font_title.render("Virtual Universe Simulator", True, THEME.accent)
        self.screen.blit(title, (16, 12))
        info = (
            f"{self.universe.name}   |   Sim: {fmt_time_days(self.universe.sim_time)}   |   "
            f"Real: {self.universe.real_elapsed:.1f}s   |   "
            f"N={len(self.universe.live_bodies())}   |   "
            f"Mode: {self.camera.mode.upper()}   |   "
            f"{'▶' if self.universe.running else '⏸'} {self.universe.speed_multiplier:g}×"
        )
        img = self.font_small.render(info, True, THEME.text_dim)
        self.screen.blit(img, (320, 14))

    def _draw_universe_view(self) -> None:
        view = self._view_rect()
        # Draw into subsurface
        surf = self.screen.subsurface(view)
        self.renderer.draw(surf, self.universe, self.camera, self.universe.selected_id)
        pygame.draw.rect(self.screen, THEME.panel_border, view, 1)

        # Learn mode overlay
        if self.universe.learn_mode:
            selected = self.universe.get_body(self.universe.selected_id or "")
            msgs = learn_messages(self.universe, selected)
            y = view.y + 10
            for m in msgs[:3]:
                bubble = self.font_small.render(m[:110], True, THEME.accent2)
                bg = pygame.Surface((bubble.get_width() + 16, bubble.get_height() + 10), pygame.SRCALPHA)
                bg.fill((10, 20, 40, 180))
                self.screen.blit(bg, (view.x + 12, y))
                self.screen.blit(bubble, (view.x + 20, y + 5))
                y += bubble.get_height() + 16

        if self.universe.diagnostics.energy_drift_warning:
            warn = self.font_small.render(
                "⚠ Energy drift significant — reduce dt or avoid close encounters",
                True,
                THEME.warning,
            )
            self.screen.blit(warn, (view.x + 12, view.bottom - 28))

    def _draw_bottom_bar(self) -> None:
        bar = self._bottom_rect()
        pygame.draw.rect(self.screen, THEME.panel, bar)
        pygame.draw.line(self.screen, THEME.panel_border, (0, bar.y), (bar.width, bar.y))
        x = 12
        y = bar.y + 12
        btn_h = 32
        gap = 8

        def btn(label, w=70):
            nonlocal x
            r = pygame.Rect(x, y, w, btn_h)
            clicked = self.ui.button(self.screen, r, label)
            x += w + gap
            return clicked

        if btn("▶ Play"):
            self.universe.play()
        if btn("⏸ Pause"):
            self.universe.pause()
        if btn("⏭ Step"):
            self.universe.pause()
            self.universe.step_once()
        if btn("⏮ Reset"):
            self.universe.reset_to_snapshot()
            self.status_message = "Reset to initial conditions."

        # Speed
        if btn("Speed −", 80):
            self.speed_index = max(0, self.speed_index - 1)
            self.universe.set_speed(SPEEDS[self.speed_index])
        speed_label = f"{SPEEDS[self.speed_index]:g}×"
        self.ui.label(self.screen, (x, y + 8), speed_label)
        x += 70
        if btn("Speed +", 80):
            self.speed_index = min(len(SPEEDS) - 1, self.speed_index + 1)
            self.universe.set_speed(SPEEDS[self.speed_index])

        if btn("Solar Sys", 90):
            load_solar_system(self.universe)
            self.camera.fit_bodies(self.universe.bodies, self._view_size())
            self.universe.play()
            self.status_message = "Loaded Solar System."
        if btn("Reset SS", 90):
            if self.universe.name == "Solar System":
                self.universe.reset_to_snapshot()
            else:
                load_solar_system(self.universe)
                self.camera.fit_bodies(self.universe.bodies, self._view_size())
            self.status_message = "Solar System restored."

        # Scenario cycle
        if btn("◀ Scen", 70):
            self.scenario_index = (self.scenario_index - 1) % len(self.scenario_names)
        scen = self.scenario_names[self.scenario_index]
        short = scen if len(scen) < 22 else scen[:20] + "…"
        self.ui.label(self.screen, (x, y + 8), short, small=True)
        x += 160
        if btn("Scen ▶", 70):
            self.scenario_index = (self.scenario_index + 1) % len(self.scenario_names)
        if btn("Load", 60):
            load_scenario(self.universe, self.scenario_names[self.scenario_index])
            self.camera.fit_bodies(self.universe.bodies, self._view_size())
            self.universe.play()
            self.status_message = f"Loaded: {self.scenario_names[self.scenario_index]}"

        if btn("2D/3D", 70):
            self.camera.mode = "3d" if self.camera.mode == "2d" else "2d"
        if btn("Fit", 50):
            self.camera.fit_bodies(self.universe.bodies, self._view_size())

        # Status
        st = self.font_small.render(self.status_message[:60], True, THEME.text_dim)
        self.screen.blit(st, (bar.right - st.get_width() - PANEL_W - 12, y + 8))

    def _draw_side_panel(self) -> None:
        panel = self._panel_rect()
        pygame.draw.rect(self.screen, THEME.panel, panel)
        pygame.draw.line(
            self.screen, THEME.panel_border, (panel.x, panel.y), (panel.x, panel.bottom)
        )

        # Tabs
        tabs = [
            ("object", "Object"),
            ("builder", "Builder"),
            ("physics", "Physics"),
            ("events", "Events"),
            ("graphs", "Graphs"),
            ("about", "About"),
        ]
        tx = panel.x + 6
        ty = panel.y + 8
        for key, label in tabs:
            r = pygame.Rect(tx, ty, 50, 22)
            if self.ui.button(self.screen, r, label, active=(self.right_tab == key)):
                self.right_tab = key
                if key == "object":
                    self._sync_edit_fields()
            tx += 54

        content = pygame.Rect(panel.x + 10, panel.y + 40, panel.width - 20, panel.height - 50)
        if self.right_tab == "object":
            self._draw_object_panel(content)
        elif self.right_tab == "builder":
            self._draw_builder_panel(content)
        elif self.right_tab == "physics":
            self._draw_physics_panel(content)
        elif self.right_tab == "events":
            self._draw_events_panel(content)
        elif self.right_tab == "graphs":
            self._draw_graphs_panel(content)
        else:
            self._draw_about_panel(content)

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    def _selected(self) -> Optional[Body]:
        if not self.universe.selected_id:
            return None
        return self.universe.get_body(self.universe.selected_id)

    def _sync_edit_fields(self) -> None:
        b = self._selected()
        if b is None:
            self.edit_fields = {}
            return
        self.edit_fields = {
            "name": b.name,
            "mass": f"{b.mass:.6e}",
            "radius": f"{b.radius:.6e}",
            "px": f"{b.position[0]:.6e}",
            "py": f"{b.position[1]:.6e}",
            "pz": f"{b.position[2]:.6e}",
            "vx": f"{b.velocity[0]:.6e}",
            "vy": f"{b.velocity[1]:.6e}",
            "vz": f"{b.velocity[2]:.6e}",
        }

    def _apply_edit_fields(self) -> None:
        b = self._selected()
        if b is None:
            return
        try:
            b.name = self.edit_fields.get("name", b.name)
            b.mass = float(self.edit_fields.get("mass", b.mass))
            b.radius = float(self.edit_fields.get("radius", b.radius))
            b.position = np.array(
                [
                    float(self.edit_fields["px"]),
                    float(self.edit_fields["py"]),
                    float(self.edit_fields["pz"]),
                ]
            )
            b.velocity = np.array(
                [
                    float(self.edit_fields["vx"]),
                    float(self.edit_fields["vy"]),
                    float(self.edit_fields["vz"]),
                ]
            )
            self.status_message = f"Updated {b.name}."
            self.universe.log_event(f"User edited properties of {b.name}.", "info")
        except (ValueError, KeyError):
            self.status_message = "Invalid numeric field — check scientific notation."

    def _draw_object_panel(self, rect: pygame.Rect) -> None:
        y = rect.y
        x = rect.x
        # View toggles
        self.ui.label(self.screen, (x, y), "Display", dim=True, small=True)
        y += 18
        self.renderer.show_labels = self.ui.toggle(
            self.screen, pygame.Rect(x, y, 200, 20), "Labels", self.renderer.show_labels
        )
        y += 22
        self.renderer.show_trails = self.ui.toggle(
            self.screen, pygame.Rect(x, y, 200, 20), "Orbital trails", self.renderer.show_trails
        )
        y += 22
        self.renderer.show_velocity = self.ui.toggle(
            self.screen, pygame.Rect(x, y, 200, 20), "Velocity vectors", self.renderer.show_velocity
        )
        y += 22
        self.renderer.show_gravity = self.ui.toggle(
            self.screen, pygame.Rect(x, y, 200, 20), "Gravity vectors", self.renderer.show_gravity
        )
        y += 22
        self.renderer.show_grid = self.ui.toggle(
            self.screen, pygame.Rect(x, y, 200, 20), "Coordinate grid", self.renderer.show_grid
        )
        y += 22
        self.universe.learn_mode = self.ui.toggle(
            self.screen, pygame.Rect(x, y, 200, 20), "Learn Mode", self.universe.learn_mode
        )
        y += 28

        b = self._selected()
        if b is None:
            self.ui.label(self.screen, (x, y), "Click a body to inspect.", dim=True)
            y += 30
        else:
            self.ui.label(self.screen, (x, y), "Selected Object", dim=True, small=True)
            y += 18
            parent = self.universe.get_body(b.parent_id) if b.parent_id else None
            lines = [
                f"Name: {b.name}",
                f"Type: {b.body_type.value}",
                f"Mass: {fmt_mass(b.mass)}",
                f"Radius: {sci(b.radius, 'm')}",
                f"Pos: ({sci(b.position[0])}, {sci(b.position[1])}, {sci(b.position[2])}) m",
                f"Vel: ({sci(b.velocity[0])}, {sci(b.velocity[1])}, {sci(b.velocity[2])}) m/s",
                f"Speed: {fmt_speed(b.speed)}",
                f"Accel: {sci(float(np.linalg.norm(b.acceleration)), 'm/s²')}",
                f"KE: {sci(b.kinetic_energy, 'J')}",
            ]
            if parent:
                pe = self.universe.potential_with(b, parent)
                lines.append(f"PE (w/ {parent.name}): {sci(pe, 'J')}")
                lines.append(f"Distance: {fmt_distance(b.distance_to(parent))}")
                orb = analyze_orbit(b, parent, G_const=self.universe.G)
                lines.append(f"Bound: {orb.bound}")
                if orb.eccentricity is not None:
                    lines.append(f"Eccentricity: {orb.eccentricity:.4f}")
                if orb.period:
                    lines.append(f"Period: {format_period(orb.period)}")
                if orb.periapsis and orb.apoapsis:
                    lines.append(f"Periapsis: {fmt_distance(orb.periapsis)}")
                    lines.append(f"Apoapsis: {fmt_distance(orb.apoapsis)}")
                lines.append(f"Orbital energy ε: {sci(orb.specific_energy, 'J/kg')}")
            for line in lines:
                self.ui.label(self.screen, (x, y), line[:48], small=True)
                y += 16

            y += 6
            if self.ui.button(self.screen, pygame.Rect(x, y, 140, 26), "Focus camera"):
                self.universe.focus_id = b.body_id
                self.camera.follow_id = b.body_id
            if self.ui.button(self.screen, pygame.Rect(x + 150, y, 140, 26), "Unfocus"):
                self.universe.focus_id = None
                self.camera.follow_id = None
            y += 34

        # Quick velocity experiment
        self.ui.label(self.screen, (x, y), "Quick Experiment", dim=True, small=True)
        y += 18
        if b is not None:
            if self.ui.button(self.screen, pygame.Rect(x, y, 100, 26), "v × 0.8"):
                b.velocity *= 0.8
                self.universe.log_event(f"{b.name} velocity × 0.8", "info")
                self._sync_edit_fields()
            if self.ui.button(self.screen, pygame.Rect(x + 110, y, 100, 26), "v × 1.2"):
                b.velocity *= 1.2
                self.universe.log_event(f"{b.name} velocity × 1.2", "info")
                self._sync_edit_fields()
            y += 30
            if self.ui.button(self.screen, pygame.Rect(x, y, 100, 26), "Mass × 2"):
                b.mass *= 2
                self._sync_edit_fields()
            if self.ui.button(self.screen, pygame.Rect(x + 110, y, 100, 26), "Mass × 0.5"):
                b.mass *= 0.5
                self._sync_edit_fields()
            y += 36

        # Export
        self.ui.label(self.screen, (x, y), "Export", dim=True, small=True)
        y += 18
        if self.ui.button(self.screen, pygame.Rect(x, y, 140, 26), "Export CSV"):
            path = self.export_dir / f"snapshot_{int(self.universe.sim_time)}.csv"
            export_csv(self.universe, path)
            self.status_message = f"Wrote {path.name}"
        if self.ui.button(self.screen, pygame.Rect(x + 150, y, 150, 26), "Export Report"):
            path = self.export_dir / f"report_{int(self.universe.sim_time)}.txt"
            export_experiment_report(self.universe, path)
            self.status_message = f"Wrote {path.name}"

    def _draw_builder_panel(self, rect: pygame.Rect) -> None:
        x, y = rect.x, rect.y
        self.ui.label(self.screen, (x, y), "Universe Builder", dim=True, small=True)
        y += 20

        b = self._selected()
        fields = [
            ("name", "Name"),
            ("mass", "Mass (kg)"),
            ("radius", "Radius (m)"),
            ("px", "Pos X"),
            ("py", "Pos Y"),
            ("pz", "Pos Z"),
            ("vx", "Vel X"),
            ("vy", "Vel Y"),
            ("vz", "Vel Z"),
        ]
        if not self.edit_fields and b is not None:
            self._sync_edit_fields()

        for key, label in fields:
            self.ui.label(self.screen, (x, y), label, small=True, dim=True)
            y += 14
            val = self.edit_fields.get(key, "")
            new = self.ui.text_input(
                self.screen, pygame.Rect(x, y, rect.width, 24), f"edit_{key}", val
            )
            self.edit_fields[key] = new
            y += 28

        if self.ui.button(self.screen, pygame.Rect(x, y, 100, 26), "Apply"):
            self._apply_edit_fields()
        if self.ui.button(self.screen, pygame.Rect(x + 110, y, 100, 26), "Duplicate"):
            if self.universe.selected_id:
                clone = self.universe.duplicate_body(self.universe.selected_id)
                if clone:
                    self.universe.selected_id = clone.body_id
                    self._sync_edit_fields()
        if self.ui.button(self.screen, pygame.Rect(x + 220, y, 90, 26), "Delete"):
            if self.universe.selected_id:
                self.universe.remove_body(self.universe.selected_id)
                self.universe.selected_id = None
                self.edit_fields = {}
        y += 34

        # Type cycle for selected
        if b is not None:
            if self.ui.button(self.screen, pygame.Rect(x, y, 200, 26), f"Type: {b.body_type.value}"):
                idx = self.body_types.index(b.body_type)
                b.body_type = self.body_types[(idx + 1) % len(self.body_types)]
            y += 34

        self.ui.label(self.screen, (x, y), "Add new body", dim=True, small=True)
        y += 18
        t = self.body_types[self.builder_type_index]
        if self.ui.button(self.screen, pygame.Rect(x, y, 200, 26), f"New type: {t.value}"):
            self.builder_type_index = (self.builder_type_index + 1) % len(self.body_types)
        y += 30
        if self.ui.button(self.screen, pygame.Rect(x, y, 140, 26), "Add at center"):
            mass = M_SUN if t == BodyType.STAR else M_EARTH
            radius = R_SUN if t == BodyType.STAR else R_EARTH
            nb = Body(
                name=f"{t.value.title()} {len(self.universe.bodies)+1}",
                mass=mass,
                radius=radius,
                position=self.camera.center.copy() + np.array([0.5 * AU, 0.0, 0.0]),
                velocity=np.array([0.0, 2e4, 0.0]),
                body_type=t,
            )
            self.universe.add_body(nb)
            self.universe.selected_id = nb.body_id
            self._sync_edit_fields()
            self.universe.save_snapshot()
        y += 36

        # Collision mode
        mode = self.collision_modes[self.collision_index]
        self.universe.collision_mode = mode
        if self.ui.button(
            self.screen, pygame.Rect(x, y, 280, 26), f"Collision: {mode.value}"
        ):
            self.collision_index = (self.collision_index + 1) % len(self.collision_modes)
            self.universe.collision_mode = self.collision_modes[self.collision_index]
        y += 36

        # Random universe
        self.ui.label(self.screen, (x, y), "Random Universe Generator", dim=True, small=True)
        y += 18
        for key, label in [("stars", "Stars"), ("planets", "Planets"), ("moons", "Moons")]:
            self.ui.label(self.screen, (x, y), label, small=True, dim=True)
            val = self.ui.text_input(
                self.screen,
                pygame.Rect(x + 70, y - 2, 80, 22),
                f"rand_{key}",
                self.random_cfg[key],
            )
            self.random_cfg[key] = val
            y += 26
        if self.ui.button(self.screen, pygame.Rect(x, y, 140, 26), "Generate"):
            try:
                load_random_universe(
                    self.universe,
                    n_stars=int(self.random_cfg["stars"]),
                    n_planets=int(self.random_cfg["planets"]),
                    n_moons=int(self.random_cfg["moons"]),
                )
                self.camera.fit_bodies(self.universe.bodies, self._view_size())
                self.universe.play()
            except ValueError:
                self.status_message = "Random config must be integers."
        if self.ui.button(self.screen, pygame.Rect(x + 150, y, 140, 26), "Randomize Again"):
            load_random_universe(
                self.universe,
                n_stars=int(self.random_cfg.get("stars") or 2),
                n_planets=int(self.random_cfg.get("planets") or 8),
                n_moons=int(self.random_cfg.get("moons") or 4),
                seed=None,
            )
            self.camera.fit_bodies(self.universe.bodies, self._view_size())

    def _draw_physics_panel(self, rect: pygame.Rect) -> None:
        x, y = rect.x, rect.y
        self.ui.label(self.screen, (x, y), "Diagnostics", dim=True, small=True)
        y += 18
        sample = None
        if self.universe.diagnostics.history:
            sample = self.universe.diagnostics.history[-1]
        if sample:
            lines = [
                f"Kinetic K: {sci(sample.kinetic, 'J')}",
                f"Potential U: {sci(sample.potential, 'J')}",
                f"Total E: {sci(sample.total_energy, 'J')}",
                f"|P|: {sci(sample.momentum_mag, 'kg·m/s')}",
                f"|L|: {sci(sample.angular_momentum_mag, 'kg·m²/s')}",
            ]
            if self.universe.diagnostics.initial_energy is not None:
                e0 = self.universe.diagnostics.initial_energy
                drift = abs(sample.total_energy - e0) / abs(e0) * 100 if abs(e0) else 0
                lines.append(f"Energy drift: {drift:.3f}%")
            for line in lines:
                self.ui.label(self.screen, (x, y), line, small=True)
                y += 16
        else:
            self.ui.label(self.screen, (x, y), "No samples yet — press Play.", dim=True)
            y += 20

        y += 10
        self.ui.label(self.screen, (x, y), "Key equations", dim=True, small=True)
        y += 18
        for eq in EQUATIONS_SHORT:
            self.ui.label(self.screen, (x, y), eq[:46], small=True)
            y += 16

        y += 10
        self.ui.label(self.screen, (x, y), f"Integrator: {self.universe.integrator.value}", small=True)
        y += 22
        if self.ui.button(self.screen, pygame.Rect(x, y, 280, 26), "Cycle integrator"):
            kinds = list(IntegratorKind)
            i = kinds.index(self.universe.integrator)
            self.universe.integrator = kinds[(i + 1) % len(kinds)]
        y += 34
        self.ui.label(self.screen, (x, y), f"dt = {self.universe.dt:.4g} s", small=True)
        y += 20
        new_dt = self.ui.slider(
            self.screen,
            pygame.Rect(x, y + 8, 280, 12),
            np.log10(max(self.universe.dt, 1.0)),
            1,
            6,
            "log10(dt)",
        )
        self.universe.dt = 10 ** new_dt

    def _draw_events_panel(self, rect: pygame.Rect) -> None:
        x, y = rect.x, rect.y
        if self.ui.button(self.screen, pygame.Rect(x, y, 120, 26), "Clear log"):
            self.universe.clear_events()
        y += 34
        events = list(reversed(self.universe.events[-40:]))
        for ev in events:
            t = fmt_time_days(ev.sim_time)
            color = THEME.text
            if ev.category == "collision":
                color = THEME.danger
            elif ev.category == "warning":
                color = THEME.warning
            elif ev.category == "orbit":
                color = THEME.accent2
            line = f"[{t}] {ev.message}"
            img = self.font_small.render(line[:48], True, color)
            self.screen.blit(img, (x, y))
            y += 16
            if y > rect.bottom - 10:
                break

    def _draw_graphs_panel(self, rect: pygame.Rect) -> None:
        x, y = rect.x, rect.y
        self.ui.label(self.screen, (x, y), "Energy & Momentum vs Time", dim=True, small=True)
        y += 22
        series = self.universe.diagnostics.series()
        graph_rect = pygame.Rect(x, y, rect.width, 220)
        pygame.draw.rect(self.screen, THEME.bg, graph_rect, border_radius=6)
        pygame.draw.rect(self.screen, THEME.panel_border, graph_rect, 1, border_radius=6)
        self._plot_series(graph_rect, series)
        y = graph_rect.bottom + 16
        self.ui.label(self.screen, (x, y), "Legend:", small=True, dim=True)
        y += 18
        for label, col in [
            ("Total energy E", THEME.accent),
            ("Kinetic K", THEME.success),
            ("Potential U", THEME.warning),
            ("|P| (scaled)", (180, 140, 255)),
        ]:
            pygame.draw.line(self.screen, col, (x, y + 6), (x + 20, y + 6), 2)
            self.ui.label(self.screen, (x + 28, y), label, small=True)
            y += 18

    def _plot_series(self, rect: pygame.Rect, series: dict) -> None:
        t = series.get("t") or []
        if len(t) < 2:
            img = self.font_small.render("Awaiting data…", True, THEME.text_dim)
            self.screen.blit(img, (rect.x + 12, rect.y + 12))
            return
        t0 = t[0]
        tx = [(v - t0) for v in t]

        def draw_line(values, color, normalize=True):
            if not values:
                return
            arr = np.array(values, dtype=float)
            if normalize:
                span = np.max(np.abs(arr)) + 1e-30
                arr = arr / span
            ymin, ymax = float(np.min(arr)), float(np.max(arr))
            if abs(ymax - ymin) < 1e-30:
                ymax = ymin + 1
            pts = []
            for i, val in enumerate(arr):
                px = rect.x + 8 + (rect.width - 16) * (i / (len(arr) - 1))
                py = rect.bottom - 8 - (rect.height - 16) * ((val - ymin) / (ymax - ymin))
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.aalines(self.screen, color, False, pts)

        draw_line(series["te"], THEME.accent)
        draw_line(series["ke"], THEME.success)
        draw_line(series["pe"], THEME.warning)
        draw_line(series["p"], (180, 140, 255))

    def _draw_about_panel(self, rect: pygame.Rect) -> None:
        x, y = rect.x, rect.y
        lines = PHYSICS_ABOUT.strip().splitlines()
        start = max(0, self.about_scroll)
        for line in lines[start:]:
            self.ui.label(self.screen, (x, y), line[:48], small=True, dim=line.startswith(" "))
            y += 14
            if y > rect.bottom - 40:
                break
        if self.ui.button(self.screen, pygame.Rect(x, rect.bottom - 30, 80, 24), "Up"):
            self.about_scroll = max(0, self.about_scroll - 5)
        if self.ui.button(self.screen, pygame.Rect(x + 90, rect.bottom - 30, 80, 24), "Down"):
            self.about_scroll = min(len(lines) - 1, self.about_scroll + 5)


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
