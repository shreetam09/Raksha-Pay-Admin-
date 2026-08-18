import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, EyeOff, XCircle } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

export const AuthScreen = ({ onLogin, isDark: externalIsDark, setIsDark: externalSetIsDark }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [internalIsDark, setInternalIsDark] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const isDark = externalIsDark !== undefined ? externalIsDark : internalIsDark;
  const toggleTheme = (newValue) => {
    const nextVal = typeof newValue === 'boolean' ? newValue : !isDark;
    if (externalSetIsDark) {
      externalSetIsDark(nextVal);
    } else {
      setInternalIsDark(nextVal);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onLogin) onLogin();
  };

  return (
    <div className="min-h-screen w-full flex flex-col md:flex-row bg-[var(--color-pure-white)] dark:bg-[var(--color-obsidian)] font-sans select-none transition-colors duration-300">
      {/* Left Panel (Editorial Accent) */}
      <div className="w-full md:w-2/5 bg-[var(--color-electric-lime)] dark:bg-[var(--color-iris-gleam)] p-8 md:p-14 flex flex-col justify-between shrink-0 min-h-[320px] md:min-h-screen transition-colors duration-300">
        {/* Top Left: Brand Name */}
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-[var(--color-off-black-ink)] dark:text-white">
            RakshaPay
          </h1>
        </div>

        {/* Bottom Left: Tagline & Heading */}
        <div className="mt-12 md:mt-0">
          <p className="text-[length:var(--text-caption)] uppercase font-bold tracking-widest text-[var(--color-off-black-ink)] dark:text-white/90 mb-3 opacity-90">
            PAYMENT ELEVATED
          </p>
          <h2 className="text-4xl sm:text-5xl md:text-5xl lg:text-6xl font-bold text-[var(--color-off-black-ink)] dark:text-white leading-[1.05] tracking-tight max-w-sm">
            Make your payment safe
          </h2>
        </div>
      </div>

      {/* Right Panel (Workspace) */}
      <div className="w-full md:w-3/5 bg-[var(--color-pure-white)] dark:bg-[var(--color-obsidian)] relative flex items-center justify-center p-6 sm:p-10 md:p-12 min-h-screen transition-colors duration-300">
        {/* Theme Toggle (Top Right) */}
        <div className="absolute top-6 right-6 md:top-8 md:right-8">
          <ThemeToggle isDark={isDark} setIsDark={toggleTheme} />
        </div>

        {/* Auth Card */}
        <div className="w-full max-w-md bg-[var(--color-off-white-canvas)] dark:bg-[var(--color-graphite-dark)] rounded-[var(--radius-3xl)] p-8 sm:p-10 md:p-12 shadow-sm border border-[var(--color-ash)]/40 dark:border-[var(--color-steel)] my-auto transition-colors duration-300">
          <AnimatePresence mode="wait">
            <motion.div
              key={isLogin ? 'login' : 'signup'}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              className="w-full"
            >
              {/* Header */}
              <div className="mb-6">
                <h2 className="text-[length:var(--text-heading)] font-semibold text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] tracking-tight">
                  {isLogin ? 'Welcome back' : 'Welcome'}
                </h2>
                <p className="text-[length:var(--text-body-sm)] text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] mt-1">
                  {isLogin ? 'Sign in to your account' : 'Sign up for new account'}
                </p>
              </div>

              {/* Google Button */}
              <button
                type="button"
                onClick={() => onLogin && onLogin()}
                className="w-full bg-transparent border border-[var(--color-ash)] dark:border-[var(--color-steel)] rounded-[var(--radius-full)] py-2.5 px-4 flex items-center justify-center gap-2.5 text-[length:var(--text-body-sm)] font-medium text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] hover:bg-[var(--color-pure-white)]/60 dark:hover:bg-white/10 transition-colors cursor-pointer"
              >
                <XCircle className="w-4 h-4 text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] shrink-0" />
                <span>{isLogin ? 'Sign in with Google' : 'Sign up with Google'}</span>
              </button>

              {/* Divider */}
              <div className="flex items-center my-6">
                <div className="flex-1 border-t border-[var(--color-ash)] dark:border-[var(--color-steel)]" />
                <span className="px-3 text-[length:var(--text-caption)] text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)]">or</span>
                <div className="flex-1 border-t border-[var(--color-ash)] dark:border-[var(--color-steel)]" />
              </div>

              {/* Form */}
              <form onSubmit={handleSubmit} className="space-y-4">
                {/* Email Input */}
                <div>
                  <label 
                    htmlFor="email" 
                    className="block text-[length:var(--text-caption)] font-semibold text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] mb-1.5"
                  >
                    Email address
                  </label>
                  <input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="name@domain.com"
                    className="w-full bg-[var(--color-pure-white)] dark:bg-[var(--color-abyss)] border border-[var(--color-ash)] dark:border-[var(--color-steel)] rounded-[var(--radius-md)] px-3.5 py-2.5 text-[length:var(--text-body-sm)] text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] placeholder-[var(--color-graphite)]/60 dark:placeholder-[var(--color-ash-dark)]/60 focus:outline-none focus:border-[var(--color-off-black-ink)] dark:focus:border-[var(--color-iris-gleam)] transition-colors"
                  />
                </div>

                {/* Password Input */}
                <div>
                  <label 
                    htmlFor="password" 
                    className="block text-[length:var(--text-caption)] font-semibold text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] mb-1.5"
                  >
                    Password
                  </label>
                  <div className="relative flex items-center">
                    <input
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="............"
                      className="w-full bg-[var(--color-pure-white)] dark:bg-[var(--color-abyss)] border border-[var(--color-ash)] dark:border-[var(--color-steel)] rounded-[var(--radius-md)] pl-3.5 pr-10 py-2.5 text-[length:var(--text-body-sm)] text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] placeholder-[var(--color-graphite)]/60 dark:placeholder-[var(--color-ash-dark)]/60 focus:outline-none focus:border-[var(--color-off-black-ink)] dark:focus:border-[var(--color-iris-gleam)] transition-colors"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] hover:text-[var(--color-off-black-ink)] dark:hover:text-white transition-colors cursor-pointer p-1"
                      aria-label={showPassword ? "Hide password" : "Show password"}
                    >
                      {showPassword ? (
                        <Eye className="w-4 h-4 shrink-0" />
                      ) : (
                        <EyeOff className="w-4 h-4 shrink-0" />
                      )}
                    </button>
                  </div>

                  {/* Forgot Password (ONLY if isLogin === true) */}
                  {isLogin && (
                    <div className="text-right mt-1.5">
                      <a 
                        href="#forgot-password" 
                        className="text-[length:var(--text-caption)] text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] hover:text-[var(--color-off-black-ink)] dark:hover:text-white underline transition-colors cursor-pointer"
                      >
                        Forgot password?
                      </a>
                    </div>
                  )}
                </div>

                {/* Primary Action Button */}
                <button
                  type="submit"
                  className="w-full bg-[var(--color-electric-lime)] dark:bg-[var(--color-iris-gleam)] text-[var(--color-off-black-ink)] dark:text-white font-bold text-[length:var(--text-body-sm)] py-3 rounded-[var(--radius-full)] hover:opacity-90 active:scale-[0.99] transition-all cursor-pointer shadow-sm mt-6"
                >
                  {isLogin ? 'Sign in' : 'Sign up'}
                </button>
              </form>

              {/* Footer Toggle */}
              <div className="text-center mt-6 text-[length:var(--text-caption)] text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)]">
                <span>
                  {isLogin ? "Don't have an account? " : "Already have an account? "}
                </span>
                <button
                  type="button"
                  onClick={() => setIsLogin(!isLogin)}
                  className="font-bold underline text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] cursor-pointer hover:opacity-80 transition-opacity ml-0.5"
                >
                  {isLogin ? 'Sign up' : 'Sign in'}
                </button>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default AuthScreen;