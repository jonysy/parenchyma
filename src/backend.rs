use {ContextImp, Device, Error, Framework};

pub struct Backend<F> { cx: ContextImp, framework: F }

impl<F> Backend<F> where F: Framework {

    /// Initializes a `Backend`.
    ///
    /// # Arguments 
    ///
    pub fn new(framework: F, devices: &[Device<F::Context>]) -> Result<Backend<F>, Error> {

        let cx = framework.new_context(devices)?;

        Ok(Backend {
        	cx: cx,
        	framework: framework,
        })
    }

    pub fn default() -> Result<Backend<F>, Error> {
    	let f = F::new();
    	let cx = f.new_context(f.devices())?;

    	Ok(Backend {
    		cx: cx,
    		framework: f,
    	})
    }
}