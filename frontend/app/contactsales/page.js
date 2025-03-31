"use client";

import { useState } from "react";
import { 
  IconMail, 
  IconPhone, 
  IconMapPin, 
  IconSend, 
  IconArrowRight, 
  IconBrandLinkedin, 
  IconBuildingSkyscraper, 
  IconWorld, 
  IconCheck 
} from "@tabler/icons-react";

export default function ContactSales() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
    message: "",
  });
  
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form Submitted:", formData);
    setSubmitted(true);
    
    // Reset the form after showing success message
    setTimeout(() => {
      setFormData({ 
        name: "", 
        email: "", 
        company: "",
        message: "" 
      });
      setSubmitted(false);
    }, 5000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[rgb(10,10,10)] to-[rgb(15,15,18)] text-white">
      {/* Hero Section */}
      <div className="w-full bg-[rgb(12,12,15)] border-b border-gray-800/30">
        <div className="max-w-7xl mx-auto px-6 py-16 md:py-24">
          <div className="max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
              Let's Talk Business
            </h1>
            <p className="text-xl text-gray-400 mb-8">
              Connect with our sales team to explore how VisionFlow can transform your operations.
            </p>
            
            <div className="flex flex-wrap gap-4 mt-8">
              <div className="flex items-center gap-2 bg-gray-800/40 px-4 py-2 rounded-full">
                <IconCheck size={16} className="text-gray-400" />
                <span className="text-gray-300 text-sm">Custom Solutions</span>
              </div>
              <div className="flex items-center gap-2 bg-gray-800/40 px-4 py-2 rounded-full">
                <IconCheck size={16} className="text-gray-400" />
                <span className="text-gray-300 text-sm">Expert Support</span>
              </div>
              <div className="flex items-center gap-2 bg-gray-800/40 px-4 py-2 rounded-full">
                <IconCheck size={16} className="text-gray-400" />
                <span className="text-gray-300 text-sm">Enterprise Scale</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-10">
          {/* Left Column - Contact Info */}
          <div className="lg:col-span-2 space-y-8">
            <div className="bg-gradient-to-br from-[rgb(18,18,22)] to-[rgb(15,15,18)] p-8 rounded-2xl border border-gray-800/50 shadow-lg">
              <h3 className="text-2xl font-bold mb-6">Get In Touch</h3>
              
              <div className="space-y-6">
                <div className="flex items-start space-x-4 group">
                  <div className="p-3 bg-gray-800/60 rounded-xl group-hover:bg-gray-700/80 transition-colors">
                    <IconMail className="text-gray-300" size={22} />
                  </div>
                  <div>
                    <p className="font-medium text-gray-200 group-hover:text-white transition-colors">Email Us</p>
                    <a href="mailto:visionflow.business@gmail.com" className="text-gray-400 hover:text-gray-300 transition-colors cursor-pointer">
                    visionflow.business@gmail.com</a>
                    <p className="text-gray-500 text-sm">We respond within 48 hours</p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-4 group">
                  <div className="p-3 bg-gray-800/60 rounded-xl group-hover:bg-gray-700/80 transition-colors">
                    <IconPhone className="text-gray-300" size={22} />
                  </div>
                  <div>
                    <p className="font-medium text-gray-200 group-hover:text-white transition-colors">Call Us</p>
                    <p className="text-gray-400 hover:text-gray-300 transition-colors cursor-pointer">
                      +91 98765 43210
                    </p>
                    <p className="text-gray-500 text-sm">Mon-Fri, 9AM-6PM IST</p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-4 group">
                  <div className="p-3 bg-gray-800/60 rounded-xl group-hover:bg-gray-700/80 transition-colors">
                    <IconMapPin className="text-gray-300" size={22} />
                  </div>
                  <div>
                    <p className="font-medium text-gray-200 group-hover:text-white transition-colors">Visit Us</p>
                    <p className="text-gray-400">
                      Gujarat, India
                    </p>
                    <p className="text-gray-500 text-sm">
                      By appointment only
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="mt-8 pt-8 border-t border-gray-800/50">
                <p className="text-sm text-gray-400 mb-4">
                  Connect with us on social media
                </p>
                <div className="flex space-x-3">
                  <a href="#" className="p-2 bg-gray-800/40 hover:bg-gray-700/60 rounded-lg transition-colors">
                    <IconBrandLinkedin size={20} className="text-gray-400" />
                  </a>
                  <a href="#" className="p-2 bg-gray-800/40 hover:bg-gray-700/60 rounded-lg transition-colors">
                    <IconWorld size={20} className="text-gray-400" />
                  </a>
                  <a href="#" className="p-2 bg-gray-800/40 hover:bg-gray-700/60 rounded-lg transition-colors">
                    <IconBuildingSkyscraper size={20} className="text-gray-400" />
                  </a>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-[rgb(18,18,22)] to-[rgb(15,15,18)] p-8 rounded-2xl border border-gray-800/50 shadow-lg">
              <h3 className="text-xl font-bold mb-4">Why Choose VisionFlow?</h3>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="mt-1 text-gray-400">
                    <IconCheck size={16} />
                  </div>
                  <p className="text-gray-300 text-sm">
                    Industry-leading AI technology with 99.8% accuracy
                  </p>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="mt-1 text-gray-400">
                    <IconCheck size={16} />
                  </div>
                  <p className="text-gray-300 text-sm">
                    Flexible pricing models tailored to your business needs
                  </p>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="mt-1 text-gray-400">
                    <IconCheck size={16} />
                  </div>
                  <p className="text-gray-300 text-sm">
                    Dedicated support team with 24/7 availability
                  </p>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="mt-1 text-gray-400">
                    <IconCheck size={16} />
                  </div>
                  <p className="text-gray-300 text-sm">
                    Seamless integration with your existing workflows
                  </p>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right Column - Contact Form */}
          <div className="lg:col-span-3">
            <div className="bg-gradient-to-br from-[rgb(18,18,22)] to-[rgb(15,15,18)] p-8 rounded-2xl border border-gray-800/50 shadow-lg h-full">
              {submitted ? (
                <div className="flex flex-col items-center justify-center h-full py-12">
                  <div className="w-20 h-20 bg-gray-800/40 rounded-full flex items-center justify-center mb-6">
                    <IconSend size={36} className="text-gray-300" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4">Message Sent!</h3>
                  <p className="text-gray-400 text-center max-w-md mb-6">
                    Thank you for reaching out. Our sales team will contact you shortly to discuss how we can help with your business needs.
                  </p>
                  <div className="text-sm text-gray-500">
                    Reference: VF-{Math.floor(Math.random() * 10000)}
                  </div>
                </div>
              ) : (
                <>
                  <h3 className="text-2xl font-bold mb-2">Contact Our Sales Team</h3>
                  <p className="text-gray-400 mb-8">
                    Fill out the form below and we'll get back to you within 24 hours.
                  </p>
                  
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-gray-300 mb-2 text-sm">Full Name*</label>
                        <input
                          type="text"
                          name="name"
                          value={formData.name}
                          onChange={handleChange}
                          required
                          placeholder="John Smith"
                          className="w-full bg-[rgb(22,22,26)] p-4 rounded-xl focus:ring-2 focus:ring-gray-700 outline-none border border-gray-800/50 transition-all"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-gray-300 mb-2 text-sm">Email Address*</label>
                        <input
                          type="email"
                          name="email"
                          value={formData.email}
                          onChange={handleChange}
                          required
                          placeholder="john@company.com"
                          className="w-full bg-[rgb(22,22,26)] p-4 rounded-xl focus:ring-2 focus:ring-gray-700 outline-none border border-gray-800/50 transition-all"
                        />
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-gray-300 mb-2 text-sm">Company Name</label>
                      <input
                        type="text"
                        name="company"
                        value={formData.company}
                        onChange={handleChange}
                        placeholder="Your company name"
                        className="w-full bg-[rgb(22,22,26)] p-4 rounded-xl focus:ring-2 focus:ring-gray-700 outline-none border border-gray-800/50 transition-all"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-gray-300 mb-2 text-sm">How can we help?*</label>
                      <textarea
                        name="message"
                        value={formData.message}
                        onChange={handleChange}
                        required
                        rows="5"
                        placeholder="Please describe your business needs and how we can assist you..."
                        className="w-full bg-[rgb(22,22,26)] p-4 rounded-xl focus:ring-2 focus:ring-gray-700 outline-none border border-gray-800/50 transition-all resize-none"
                      />
                    </div>
                    
                    <div className="bg-gray-800/20 p-4 rounded-xl border border-gray-800/30">
                      <div className="flex items-start">
                        <input
                          type="checkbox"
                          id="privacy"
                          className="mt-1 bg-gray-800 border-gray-700 rounded"
                        />
                        <label htmlFor="privacy" className="ml-3 text-sm text-gray-400">
                          I agree to the privacy policy and consent to being contacted regarding my request. 
                          Your information will never be shared with third parties.
                        </label>
                      </div>
                    </div>
                    
                    <button
                      type="submit"
                      className="w-full bg-gray-800 hover:bg-gray-700 transition-all p-4 rounded-xl flex items-center justify-center space-x-3 group border border-gray-700/50 shadow-lg hover:shadow-gray-900/10"
                    >
                      <span>Send Message</span>
                      <IconArrowRight size={18} className="transition-transform group-hover:translate-x-1" />
                    </button>
                  </form>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* FAQ Section */}
      <div className="w-full bg-[rgb(12,12,15)] border-t border-gray-800/30 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-2xl font-bold mb-8 text-center">Frequently Asked Questions</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            <div className="bg-[rgb(15,15,18)] p-6 rounded-xl border border-gray-800/50">
              <h3 className="font-bold mb-2">What industries do you serve?</h3>
              <p className="text-gray-400 text-sm">
                VisionFlow serves a wide range of industries including healthcare, finance, retail, manufacturing, and more. Our solutions are customizable to meet industry-specific requirements.
              </p>
            </div>
            
            <div className="bg-[rgb(15,15,18)] p-6 rounded-xl border border-gray-800/50">
              <h3 className="font-bold mb-2">How quickly can we implement your solution?</h3>
              <p className="text-gray-400 text-sm">
                Most clients are up and running within 2-4 weeks, depending on the complexity of integration. Our team works closely with you to ensure a smooth implementation process.
              </p>
            </div>
            
            <div className="bg-[rgb(15,15,18)] p-6 rounded-xl border border-gray-800/50">
              <h3 className="font-bold mb-2">Do you offer custom solutions?</h3>
              <p className="text-gray-400 text-sm">
                Yes, we specialize in tailoring our platform to your specific business needs. Our team will work with you to customize features, integrations, and workflows.
              </p>
            </div>
            
            <div className="bg-[rgb(15,15,18)] p-6 rounded-xl border border-gray-800/50">
              <h3 className="font-bold mb-2">What support options are available?</h3>
              <p className="text-gray-400 text-sm">
                We offer multiple tiers of support, from standard business hours to 24/7 premium support with dedicated account managers for enterprise clients.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}