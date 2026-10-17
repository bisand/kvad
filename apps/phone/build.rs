fn main() {
    // Cupertino: Slint's widgets drawn the way iOS draws its own.
    let config = slint_build::CompilerConfiguration::new().with_style("cupertino".into());
    slint_build::compile_with_config("ui/app.slint", config).expect("ui/app.slint compiles");
}
