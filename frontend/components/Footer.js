import React from "react";
import { Facebook, Twitter, Instagram, Linkedin } from "lucide-react";

const Footer = () => {
  return (
    <footer className="bg-transparent text-white py-8 px-6">
        <div className="border border-gray-600 w-full mb-4"></div>  
      {/* White Line Above Footer */}
      <div className="w-full h-[1px] bg-transparent mb-6"></div>

      <div className="container mx-auto text-center">
        {/* Logo or Title */}
        <h2 className="text-2xl font-bold mb-4">VisionFlow</h2>

        {/* Social Media Icons */}
        <div className="flex justify-center space-x-6 mb-6">
          <a
            href="https://facebook.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-blue-400 transition"
          >
            <Facebook className="w-6 h-6" />
          </a>
          <a
            href="https://twitter.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-blue-400 transition"
          >
            <Twitter className="w-6 h-6" />
          </a>
          <a
            href="https://instagram.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-blue-400 transition"
          >
            <Instagram className="w-6 h-6" />
          </a>
          <a
            href="https://linkedin.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-blue-500 transition"
          >
            <Linkedin className="w-6 h-6" />
          </a>
        </div>

        {/* Links */}
        <div className="flex justify-center space-x-8 mb-6 text-sm">
          <a
            href="#about"
            className="text-gray-400 hover:text-white transition"
          >
            About Us
          </a>
          <a
            href="#services"
            className="text-gray-400 hover:text-white transition"
          >
            Services
          </a>
          <a
            href="#contact"
            className="text-gray-400 hover:text-white transition"
          >
            Contact
          </a>
          <a
            href="#privacy"
            className="text-gray-400 hover:text-white transition"
          >
            Privacy Policy
          </a>
        </div>

        {/* Copyright */}
        <p className="text-gray-500 text-sm">
          © {new Date().getFullYear()} VisionFlow. All rights reserved.
        </p>
      </div>
    </footer>
  );
};

export default Footer;
