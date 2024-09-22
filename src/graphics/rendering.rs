pub trait Renderer {
    fn clear(&mut self, color: Color);
    fn draw_line(&mut self, line: &Line);
    fn present(&mut self);
}

#[cfg(target_os = "windows")]
pub struct DirectXRenderer;

#[cfg(feature = "graphics")]
pub struct VulkanRenderer;

#[cfg(target_os = "macos")]
pub struct MetalRenderer;

pub struct OpenGLRenderer;

#[cfg(target_os = "windows")]
use windows::{
    Win32::Graphics::Direct3D11::*,
    Win32::Graphics::Direct3D::*,
    Win32::Graphics::Dxgi::*,
    Win32::Foundation::*,
};

#[cfg(target_os = "macos")]
use metal::{Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLStoreAction};

use gl;
use glutin;

#[cfg(target_os = "windows")]
impl DirectXRenderer {
    pub fn new(hwnd: HWND) -> Result<Self, windows::core::Error> {
        unsafe {
            let mut feature_level = D3D_FEATURE_LEVEL_11_0;
            let mut device = None;
            let mut context = None;

            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                None,
                D3D11_CREATE_DEVICE_DEBUG,
                &[D3D_FEATURE_LEVEL_11_0],
                &mut device,
                &mut feature_level,
                &mut context,
            )?;

            let device = device.unwrap();
            let context = context.unwrap();

            let dxgi_device: IDXGIDevice = device.cast()?;
            let dxgi_adapter: IDXGIAdapter = dxgi_device.GetAdapter()?;
            let dxgi_factory: IDXGIFactory = dxgi_adapter.GetParent()?;

            let swap_chain_desc = DXGI_SWAP_CHAIN_DESC {
                BufferDesc: DXGI_MODE_DESC {
                    Width: 800,
                    Height: 600,
                    RefreshRate: DXGI_RATIONAL { Numerator: 60, Denominator: 1 },
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    ..Default::default()
                },
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: 1,
                OutputWindow: hwnd,
                Windowed: BOOL::from(true),
                SwapEffect: DXGI_SWAP_EFFECT_DISCARD,
                ..Default::default()
            };

            let swap_chain = dxgi_factory.CreateSwapChain(&device, &swap_chain_desc)?;

            let back_buffer: ID3D11Texture2D = swap_chain.GetBuffer(0)?;
            let render_target_view = device.CreateRenderTargetView(&back_buffer, None)?;

            let vs_blob = D3DCompile(VS_SRC, "vs_main", None, &[], "vs_4_0", 0, 0)?;
            let vertex_shader = device.CreateVertexShader(vs_blob.GetBufferPointer(), vs_blob.GetBufferSize(), None)?;

            let ps_blob = D3DCompile(PS_SRC, "ps_main", None, &[], "ps_4_0", 0, 0)?;
            let pixel_shader = device.CreatePixelShader(ps_blob.GetBufferPointer(), ps_blob.GetBufferSize(), None)?;

            let input_element_desc = [
                D3D11_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"POSITION\0".as_ptr()),
                    SemanticIndex: 0,
                    Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    InputSlot: 0,
                    AlignedByteOffset: 0,
                    InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                },
                D3D11_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"COLOR\0".as_ptr()),
                    SemanticIndex: 0,
                    Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    InputSlot: 0,
                    AlignedByteOffset: D3D11_APPEND_ALIGNED_ELEMENT,
                    InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                },
            ];

            let input_layout = device.CreateInputLayout(
                &input_element_desc,
                vs_blob.GetBufferPointer(),
                vs_blob.GetBufferSize(),
            )?;

            Ok(DirectXRenderer {
                device: Some(device),
                context: Some(context),
                swap_chain: Some(swap_chain),
                render_target_view: Some(render_target_view),
                vertex_shader: Some(vertex_shader),
                pixel_shader: Some(pixel_shader),
                input_layout: Some(input_layout),
            })
        }
    }
}

#[cfg(target_os = "windows")]
impl Renderer for DirectXRenderer {
    fn clear(&mut self, color: Color) {
        if let Some(context) = &self.context {
            if let Some(render_target_view) = &self.render_target_view {
                unsafe {
                    let color_rgba = [
                        color.r as f32 / 255.0,
                        color.g as f32 / 255.0,
                        color.b as f32 / 255.0,
                        color.a as f32 / 255.0,
                    ];
                    context.ClearRenderTargetView(render_target_view, &color_rgba);
                }
            }
        }
    }

    fn draw_line(&mut self, line: &Line) {
        if let (Some(device), Some(context)) = (&self.device, &self.context) {
            unsafe {
                // Crear un buffer de vértices
                let vertices = [
                    Vertex { pos: [line.start.x, line.start.y, 0.0, 1.0], color: [line.color.r as f32 / 255.0, line.color.g as f32 / 255.0, line.color.b as f32 / 255.0, line.color.a as f32 / 255.0] },
                    Vertex { pos: [line.end.x, line.end.y, 0.0, 1.0], color: [line.color.r as f32 / 255.0, line.color.g as f32 / 255.0, line.color.b as f32 / 255.0, line.color.a as f32 / 255.0] },
                ];

                let bd = D3D11_BUFFER_DESC {
                    ByteWidth: std::mem::size_of::<[Vertex; 2]>() as u32,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_VERTEX_BUFFER,
                    CPUAccessFlags: 0,
                    MiscFlags: 0,
                    StructureByteStride: 0,
                };

                let sd = D3D11_SUBRESOURCE_DATA {
                    pSysMem: vertices.as_ptr() as *const _,
                    SysMemPitch: 0,
                    SysMemSlicePitch: 0,
                };

                let mut vertex_buffer = None;
                device.CreateBuffer(&bd, Some(&sd), Some(&mut vertex_buffer)).expect("Failed to create vertex buffer");

                // Configurar el pipeline
                let stride = std::mem::size_of::<Vertex>() as u32;
                let offset = 0u32;
                context.IASetVertexBuffers(0, 1, &vertex_buffer, &stride, &offset);
                context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

                // Dibujar la línea
                context.Draw(2, 0);
            }
        }
    }

    fn present(&mut self) {
        if let Some(swap_chain) = &self.swap_chain {
            unsafe {
                swap_chain.Present(1, 0).expect("Failed to present");
            }
        }
    }
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

#[cfg(target_os = "windows")]
const VS_SRC: &[u8] = b"
struct VS_INPUT {
    float4 pos : POSITION;
    float4 color : COLOR;
};

struct PS_INPUT {
    float4 pos : SV_POSITION;
    float4 color : COLOR;
};

PS_INPUT vs_main(VS_INPUT input) {
    PS_INPUT output;
    output.pos = input.pos;
    output.color = input.color;
    return output;
}
";

#[cfg(target_os = "windows")]
const PS_SRC: &[u8] = b"
struct PS_INPUT {
    float4 pos : SV_POSITION;
    float4 color : COLOR;
};

float4 ps_main(PS_INPUT input) : SV_Target {
    return input.color;
}
";

#[cfg(feature = "graphics")]
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    device::{Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo},
    format::ClearValue,
    framebuffer::{Framebuffer, FramebufferCreateInfo, Subpass},
    image::{ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint},
    render_pass::{RenderPass, Subpass},
    swapchain::{
        AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform,
        Swapchain, SwapchainCreateInfo,
    },
    sync::{self, FlushError, GpuFuture},
    Version,
};

#[cfg(feature = "graphics")]
impl VulkanRenderer {
    pub fn new(window: &Window) -> Result<Self, Box<dyn Error>> {
        let instance = Instance::new(InstanceCreateInfo::default())?;
        
        let surface = Surface::from_window(instance.clone(), window.clone())?;
        
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        
        let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
            .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
            .filter_map(|p| {
                p.queue_families()
                    .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                    .map(|q| (p, q))
            })
            .min_by_key(|(p, _)| {
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                }
            })
            .ok_or("No se encontró un dispositivo físico adecuado")?;
        
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )?;
        
        let queue = queues.next().unwrap();
        
        let (swapchain, images) = {
            let caps = surface.capabilities(physical_device)?;
            let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;
            let dimensions: [u32; 2] = surface.window().inner_size().into();
    
            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format: format,
                    image_extent: dimensions.into(),
                    image_usage: ImageUsage::color_attachment(),
                    composite_alpha,
                    ..Default::default()
                },
            )?
        };

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )?;

        let framebuffers = images.iter().map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            ).unwrap()
        }).collect::<Vec<_>>();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: surface.window().inner_size().into(),
            depth_range: 0.0..1.0,
        };
        
        Ok(VulkanRenderer {
            instance,
            device,
            queue,
            surface,
            swapchain,
            images,
            render_pass,
            framebuffers,
            viewport,
        })
    }
}

#[cfg(feature = "graphics")]
impl Renderer for VulkanRenderer {
    fn clear(&mut self, color: Color) {
        let clear_values = vec![vulkano::format::ClearValue::Float([color.r as f32 / 255.0, color.g as f32 / 255.0, color.b as f32 / 255.0, color.a as f32 / 255.0])];

        let (image_index, _, acquire_future) = match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
            Ok(r) => r,
            Err(e) => {
                println!("Failed to acquire next image: {:?}", e);
                return;
            }
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        builder
            .begin_render_pass(
                self.framebuffers[image_index].clone(),
                SubpassContents::Inline,
                clear_values,
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = acquire_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_index)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                future.wait(None).unwrap();
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
            }
        }
    }

    fn draw_line(&mut self, line: &Line) {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            vec![
                Vertex { position: [line.start.x, line.start.y, 0.0, 1.0], color: [line.color.r as f32 / 255.0, line.color.g as f32 / 255.0, line.color.b as f32 / 255.0, line.color.a as f32 / 255.0] },
                Vertex { position: [line.end.x, line.end.y, 0.0, 1.0], color: [line.color.r as f32 / 255.0, line.color.g as f32 / 255.0, line.color.b as f32 / 255.0, line.color.a as f32 / 255.0] },
            ]
            .into_iter(),
        )
        .unwrap();

        let vs = vs::load(self.device.clone()).expect("failed to create shader module");
        let fs = fs::load(self.device.clone()).expect("failed to create shader module");

        let pipeline = GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(self.render_pass.clone(), 0).unwrap())
            .line_list()
            .build(self.device.clone())
            .unwrap();

        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.uniform_buffer.clone())]
        ).unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                self.framebuffers[self.image_index].clone(),
                SubpassContents::Inline,
                vec![[0.0, 0.0, 0.0, 1.0].into()],
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(2, 1, 0, 0)
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = self.previous_frame_end
            .take()
            .unwrap()
            .join(self.acquire_future.take().unwrap())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), self.image_index)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }

    fn present(&mut self) {
        // Verificar si necesitamos recrear el swapchain
        match self.swapchain.recreate_swapchain() {
            Ok(()) => {},
            Err(e) => {
                println!("Failed to recreate swapchain: {:?}", e);
                return;
            }
        }

        // Adquirir la siguiente imagen
        let (image_index, _, acquire_future) = match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
            Ok(r) => r,
            Err(e) => {
                println!("Failed to acquire next image: {:?}", e);
                return;
            }
        };

        // Crear un command buffer vacío
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        // Finalizar el command buffer
        let command_buffer = builder.build().unwrap();

        // Presentar la imagen
        let future = acquire_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_index)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                future.wait(None).unwrap();
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
            }
        }

        // Limpiar los recursos antiguos
        self.device.wait_idle().unwrap();
    }
}

#[cfg(feature = "graphics")]
#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 4],
    color: [f32; 4],
}
#[cfg(feature = "graphics")]
vulkano::impl_vertex!(Vertex, position, color);

#[cfg(feature = "graphics")]
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            layout(location = 0) in vec4 position;
            layout(location = 1) in vec4 color;
            layout(location = 0) out vec4 v_color;
            void main() {
                gl_Position = position;
                v_color = color;
            }
        "
    }
}

#[cfg(feature = "graphics")]
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) in vec4 v_color;
            layout(location = 0) out vec4 f_color;
            void main() {
                f_color = v_color;
            }
        "
    }
}

#[cfg(target_os = "macos")]
impl MetalRenderer {
    pub fn new(window: &winit::window::Window) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();

        // Crear el pipeline state
        let library = device.new_library_with_source(METAL_SHADER_SOURCE, &metal::CompileOptions::new())?;
        let vertex_function = library.get_function("vertex_shader", None)?;
        let fragment_function = library.get_function("fragment_shader", None)?;

        let pipeline_state_descriptor = metal::RenderPipelineDescriptor::new();
        pipeline_state_descriptor.set_vertex_function(Some(&vertex_function));
        pipeline_state_descriptor.set_fragment_function(Some(&fragment_function));
        pipeline_state_descriptor.set_vertex_descriptor(Some(&metal::VertexDescriptor::new()));

        let attachment = pipeline_state_descriptor.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let pipeline_state = device.new_render_pipeline_state(&pipeline_state_descriptor)?;

        // Crear el vertex buffer
        let vertex_buffer = device.new_buffer(
            std::mem::size_of::<[f32; 4]>() as u64 * 2,
            metal::MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        // Configurar el color attachment
        let color_attachment = metal::RenderPassColorAttachmentDescriptor::new();
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);

        Ok(MetalRenderer {
            device,
            command_queue,
            pipeline_state,
            vertex_buffer,
            color_attachment,
        })
    }
}

#[cfg(target_os = "macos")]
impl Renderer for MetalRenderer {
    fn clear(&mut self, color: Color) {
        self.color_attachment.set_clear_color(MTLClearColor::new(
            color.r as f64 / 255.0,
            color.g as f64 / 255.0,
            color.b as f64 / 255.0,
            color.a as f64 / 255.0,
        ));
    }

    fn draw_line(&mut self, line: &Line) {
        let vertices = [
            line.start.x, line.start.y, 0.0, 1.0,
            line.end.x, line.end.y, 0.0, 1.0,
        ];
        self.vertex_buffer.contents().copy_from_nonoverlapping(vertices.as_ptr() as *const u8, std::mem::size_of_val(&vertices));

        let command_buffer = self.command_queue.new_command_buffer();
        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        render_pass_descriptor.color_attachments().object_at(0).unwrap().set_texture(Some(&self.color_attachment.texture().unwrap()));

        let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        encoder.set_render_pipeline_state(&self.pipeline_state);
        encoder.set_vertex_buffer(0, Some(&self.vertex_buffer), 0);
        encoder.draw_primitives(metal::MTLPrimitiveType::Line, 0, 2);
        encoder.end_encoding();

        command_buffer.commit();
    }

    fn present(&mut self) {
        // La presentación se maneja automáticamente en Metal
    }
}

#[cfg(target_os = "macos")]
const METAL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float4 position [[attribute(0)]];
};

vertex float4 vertex_shader(VertexIn vertex_in [[stage_in]]) {
    return vertex_in.position;
}

fragment float4 fragment_shader() {
    return float4(1.0, 0.0, 0.0, 1.0);
}
"#;

impl OpenGLRenderer {
    pub fn new(window: &glutin::window::Window) -> Result<Self, Box<dyn std::error::Error>> {
        let gl_context = unsafe {
            let gl_context = glutin::ContextBuilder::new()
                .build_windowed(window.clone(), &window.display())
                .unwrap()
                .make_current()
                .unwrap();

            gl::load_with(|symbol| gl_context.get_proc_address(symbol) as *const _);

            gl_context
        };

        Ok(OpenGLRenderer { gl_context })
    }
}

impl Renderer for OpenGLRenderer {
    fn clear(&mut self, color: Color) {
        unsafe {
            gl::ClearColor(
                color.r as f32 / 255.0,
                color.g as f32 / 255.0,
                color.b as f32 / 255.0,
                color.a as f32 / 255.0,
            );
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
    }

    fn draw_line(&mut self, line: &Line) {
        unsafe {
            gl::Begin(gl::LINES);
            gl::Color4f(
                line.color.r as f32 / 255.0,
                line.color.g as f32 / 255.0,
                line.color.b as f32 / 255.0,
                line.color.a as f32 / 255.0,
            );
            gl::Vertex2f(line.start.x, line.start.y);
            gl::Vertex2f(line.end.x, line.end.y);
            gl::End();
        }
    }

    fn present(&mut self) {
        self.gl_context.swap_buffers().unwrap();
    }
}

pub enum RendererType {
    DirectX,
    Vulkan,
    Metal,
    OpenGL,
}

pub struct RenderingContext {
    renderer: Box<dyn Renderer>,
}

impl RenderingContext {
    pub fn new(renderer_type: RendererType, window: &Window) -> Result<Self, Box<dyn Error>> {
        let renderer: Box<dyn Renderer> = match renderer_type {
            RendererType::DirectX => Box::new(DirectXRenderer),
            RendererType::Vulkan => Box::new(VulkanRenderer::new(window)?),
            RendererType::Metal => Box::new(MetalRenderer::new(window)?),
            RendererType::OpenGL => Box::new(OpenGLRenderer::new(window)?),
        };
        Ok(RenderingContext { renderer })
    }

    pub fn clear(&mut self, color: Color) {
        self.renderer.clear(color);
    }

    pub fn draw_line(&mut self, line: &Line) {
        self.renderer.draw_line(line);
    }

    pub fn present(&mut self) {
        self.renderer.present();
    }

    pub fn change_renderer(&mut self, renderer_type: RendererType, window: &Window) -> Result<(), Box<dyn Error>> {
        self.renderer = match renderer_type {
            RendererType::DirectX => Box::new(DirectXRenderer),
            RendererType::Vulkan => Box::new(VulkanRenderer::new(window)?),
            RendererType::Metal => Box::new(MetalRenderer::new(window)?),
            RendererType::OpenGL => Box::new(OpenGLRenderer::new(window)?),
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::primitives::{Color, Line, Point};
    use winit::event_loop::EventLoop;
    use winit::window::WindowBuilder;

    #[test]
    fn test_rendering_context() {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new().build(&event_loop).unwrap();
        
        let mut context = RenderingContext::new(RendererType::Vulkan, &window).unwrap();
        let black = Color::black();
        let line = Line::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0), black);

        context.clear(black);
        context.draw_line(&line);
        context.present();

        context.change_renderer(RendererType::OpenGL, &window).unwrap();
        context.clear(black);
        context.draw_line(&line);
        context.present();
    }
}
