use std::sync::Arc;

use egui::mutex::RwLock;
use wgpu::{Adapter, CommandEncoder, Device, Queue, Surface, TextureView};

use crate::renderer;

/// Access to the render state for egui, which can be useful in combination with
/// [`egui::PaintCallback`]s for custom rendering using WGPU.
#[derive(Clone)]
pub struct RenderState {
    pub renderer: Arc<RwLock<renderer::Renderer>>,
}

/// Everything you need to paint egui with [`wgpu`] on [`winit`].
///
/// Alternatively you can use [`crate::renderer`] directly.
pub struct Painter {
    render_state: Option<RenderState>,
}

impl Painter {
    /// Manages [`wgpu`] state, including surface state, required to render egui.
    ///
    /// Only the [`wgpu::Instance`] is initialized here. Device selection and the initialization
    /// of render + surface state is deferred until the painter is given its first window target
    /// via [`set_window()`](Self::set_window). (Ensuring that a device that's compatible with the
    /// native window is chosen)
    ///
    /// Before calling [`paint_and_update_textures()`](Self::paint_and_update_textures) a
    /// [`wgpu::Surface`] must be initialized (and corresponding render state) by calling
    /// [`set_window()`](Self::set_window) once you have
    /// a [`winit::window::Window`] with a valid `.raw_window_handle()`
    /// associated.
    pub fn new() -> Self {
        Self { render_state: None }
    }

    /// Get the [`RenderState`].
    ///
    /// Will return [`None`] if the render state has not been initialized yet.
    pub fn render_state(&self) -> Option<RenderState> {
        self.render_state.as_ref().cloned()
    }

    fn init_render_state(
        &self,
        device: &Device,
        target_format: wgpu::TextureFormat,
    ) -> RenderState {
        let rpass = renderer::Renderer::new(&device, target_format, 1, 0);

        RenderState {
            renderer: Arc::new(RwLock::new(rpass)),
        }
    }

    // We want to defer the initialization of our render state until we have a surface
    // so we can take its format into account.
    //
    // After we've initialized our render state once though we expect all future surfaces
    // will have the same format and so this render state will remain valid.
    fn ensure_render_state_for_surface(
        &mut self,
        device: &Device,
        adapter: &Adapter,
        surface: &Surface,
    ) {
        if self.render_state.is_none() {
            let swapchain_format = surface.get_supported_formats(adapter)[0];
            let rs = self.init_render_state(device, swapchain_format);
            self.render_state = Some(rs);
        }
    }

    /// Updates (or clears) the [`winit::window::Window`] associated with the [`Painter`]
    ///
    /// This creates a [`wgpu::Surface`] for the given Window (as well as initializing render
    /// state if needed) that is used for egui rendering.
    ///
    /// This must be called before trying to render via
    /// [`paint_and_update_textures`](Self::paint_and_update_textures)
    ///
    /// # Portability
    ///
    /// _In particular it's important to note that on Android a it's only possible to create
    /// a window surface between `Resumed` and `Paused` lifecycle events, and Winit will panic on
    /// attempts to query the raw window handle while paused._
    ///
    /// On Android [`set_window`](Self::set_window) should be called with `Some(window)` for each
    /// `Resumed` event and `None` for each `Paused` event. Currently, on all other platforms
    /// [`set_window`](Self::set_window) may be called with `Some(window)` as soon as you have a
    /// valid [`winit::window::Window`].
    ///
    /// # Safety
    ///
    /// The raw Window handle associated with the given `window` must be a valid object to create a
    /// surface upon and must remain valid for the lifetime of the created surface. (The surface may
    /// be cleared by passing `None`).
    pub fn set_window(&mut self, device: &Device, adapter: &Adapter, surface: &Surface) {
        self.ensure_render_state_for_surface(device, adapter, surface);
    }

    /// Returns the maximum texture dimension supported if known
    ///
    /// This API will only return a known dimension after `set_window()` has been called
    /// at least once, since the underlying device and render state are initialized lazily
    /// once we have a window (that may determine the choice of adapter/device).
    pub fn max_texture_side(&self, device: &Device) -> Option<usize> {
        Some(device.limits().max_texture_dimension_2d as usize)
    }

    pub fn paint_and_update_textures(
        &mut self,
        pixels_per_point: f32,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: &egui::TexturesDelta,
        device: &Device,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        width: u32,
        height: u32,
        output_view: &TextureView,
    ) {
        let render_state = match self.render_state.as_mut() {
            Some(rs) => rs,
            None => return,
        };

        // Upload all resources for the GPU.
        let screen_descriptor = renderer::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };

        {
            let mut renderer = render_state.renderer.write();
            for (id, image_delta) in &textures_delta.set {
                renderer.update_texture(&device, &queue, *id, image_delta);
            }

            renderer.update_buffers(&device, &queue, clipped_primitives, &screen_descriptor);
        }

        // Record all render passes.
        render_state.renderer.read().render(
            encoder,
            &output_view,
            clipped_primitives,
            &screen_descriptor,
            None,
        );

        {
            let mut renderer = render_state.renderer.write();
            for id in &textures_delta.free {
                renderer.free_texture(id);
            }
        }
    }

    #[allow(clippy::unused_self)]
    pub fn destroy(&mut self) {
        // TODO(emilk): something here?
    }
}
