/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true, // Allows production builds to complete even with ESLint errors
  },
};

export default nextConfig;
